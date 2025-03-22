use crate::distributions::{
    Bernoulli, Categorical, Dirac, Distribution, LogLikelihood, ReDistribution, Uniform, Uniformly,
};
use log::Log;
use rand::{Rng, SeedableRng, random_iter, random_range};
use rand_pcg::Pcg64;
use std::{
    any::Any,
    collections::{BinaryHeap, HashMap},
    fmt::Debug,
    marker::PhantomData,
};
use tinyset::Set64;

#[derive(Clone, Copy, Debug)]
enum DistKind {
    Resampleable,
    Dirac,
    Observed,
}

#[derive(Debug)]
enum DistType<A> {
    Dirac(A),
    Resampleable(Box<dyn ReDistribution<A>>),
    Observed(A, Box<dyn ReDistribution<A>>),
}
impl<A: Clone> DistType<A> {
    fn sample(&self, g: &mut Pcg64) -> (A, LogLikelihood) {
        match self {
            DistType::Dirac(a) => (a.clone(), 0.0),
            DistType::Resampleable(dist) => dist.sample(g),
            DistType::Observed(a, dist) => (a.clone(), dist.log_density(a)),
        }
    }
    fn resample(
        &self,
        old_value: A,
        old_ll: LogLikelihood,
        g: &mut Pcg64,
    ) -> ((A, LogLikelihood), LogLikelihood) {
        match self {
            DistType::Dirac(_) => unreachable!(),
            DistType::Resampleable(dist) => dist.resample(old_value, old_ll, g),
            DistType::Observed(_, dist) => dist.resample(old_value, old_ll, g),
        }
    }
}
fn dirac_s<A: Clone>(a: A) -> DistType<A> {
    DistType::Dirac(a)
}
fn bern_s(p: f64) -> DistType<bool> {
    DistType::Resampleable(Box::new(Bernoulli::new(p)))
}
fn uniform_s(lower: f64, upper: f64) -> DistType<f64> {
    DistType::Resampleable(Box::new(Uniform::new(lower, upper)))
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeRef<A>(pub u32, PhantomData<A>);
impl<A> NodeRef<A> {
    pub fn new(x: u32) -> Self {
        Self(x, PhantomData)
    }
}
impl<A> Clone for NodeRef<A> {
    fn clone(&self) -> Self {
        NodeRef(self.0, PhantomData)
    }
}
impl<A> Copy for NodeRef<A> {}
impl<A> Debug for NodeRef<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("NodeRef").field(&self.0).finish()
    }
}

enum NodeChange<A> {
    ChangeValue(u64, A),
    ChangeDist(LogLikelihood, DistType<A>),
}
// to update a node
trait Updater<A> {
    fn update(
        &self,
        updated: NodeRef<A>,
        state: &MCState,
        set: &mut NodeSet,
    ) -> NodeChange<A>;
}
impl<A> Debug for dyn Updater<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Updater<{:?}>", std::any::type_name::<A>())
    }
}
struct NodeDep1<A, B> {
    dep: NodeRef<A>,
    f: fn(A) -> DistType<B>,
}
impl<A: Clone + 'static, B: 'static> Updater<B> for NodeDep1<A, B> {
    fn update(&self, updated: NodeRef<B>, state: &MCState, _: &mut NodeSet) -> NodeChange<B> {
        let now = state.now;
        assert!(
            !state.inactive.contains_key(&self.dep.0),
            "activity violation: updating the inactive. node {:?} is inactive.",
            self.dep
        );
        assert!(
            !state.inactive.contains_key(&updated.0),
            "activity violation: updating the inactive. node {:?} is inactive.",
            updated
        );

        let dep = state.get_node(self.dep);
        let this = state.get_node(updated);
        assert!(
            this.timestamp < dep.timestamp,
            "dependency violation: node {:?} depends on {:?}, but has timestamp ({}) >= ({})",
            updated,
            self.dep,
            this.timestamp,
            dep.timestamp
        );
        let dist = (self.f)(dep.value.clone());
        match dist {
            DistType::Dirac(value) => NodeChange::ChangeValue(now, value),
            DistType::Resampleable(ref d) => {
                NodeChange::ChangeDist(d.log_density(&this.value), dist)
            }
            DistType::Observed(_, ref d) => {
                NodeChange::ChangeDist(d.log_density(&this.value), dist)
            }
        }
    }
}
struct NodeDep2<A, B, C> {
    dep1: NodeRef<A>,
    dep2: NodeRef<B>,
    f: fn(A, B) -> DistType<C>,
}
impl<A: Clone + 'static, B: Clone + 'static, C: 'static> Updater<C> for NodeDep2<A, B, C> {
    fn update(&self, updated: NodeRef<C>, state: &MCState, _: &mut NodeSet) -> NodeChange<C> {
        let now = state.now;
        assert!(
            !state.inactive.contains_key(&self.dep1.0),
            "activity violation: updating the inactive. node {:?} is inactive.",
            self.dep1
        );
        assert!(
            !state.inactive.contains_key(&self.dep2.0),
            "activity violation: updating the inactive. node {:?} is inactive.",
            self.dep2
        );
        assert!(
            !state.inactive.contains_key(&updated.0),
            "activity violation: updating the inactive. node {:?} is inactive.",
            updated
        );
        let dep1 = state.get_node(self.dep1);
        let dep2 = state.get_node(self.dep2);
        let this = state.get_node(updated);
        assert!(
            this.timestamp < dep1.timestamp || this.timestamp < dep2.timestamp,
            "dependency violation: node {:?} depends on {:?} and {:?}, but has timestamp ({}) >= ({}) and ({})",
            updated,
            self.dep1,
            self.dep2,
            this.timestamp,
            dep1.timestamp,
            dep2.timestamp
        );
        let dist = (self.f)(dep1.value.clone(), dep2.value.clone());
        match dist {
            DistType::Dirac(value) => NodeChange::ChangeValue(now, value),
            DistType::Resampleable(ref d) => {
                NodeChange::ChangeDist(d.log_density(&this.value), dist)
            }
            DistType::Observed(_, ref d) => {
                NodeChange::ChangeDist(d.log_density(&this.value), dist)
            }
        }
    }
}

#[derive(Debug)]
struct Node<A> {
    value: A,
    ll: LogLikelihood,
    dist: DistType<A>,
    timestamp: u64,
    updater: Option<Box<dyn Updater<A>>>,
}
impl<A> Node<A> {
    fn resampleable(&self) -> bool {
        match self.dist {
            DistType::Resampleable(_) => true,
            _ => false,
        }
    }
}
type NodeSet = Set64<u32>;
type NodeQueue = BinaryHeap<u32>;

fn rejected_ll(ll: LogLikelihood) -> bool {
    (ll.is_infinite() && ll.is_sign_negative()) || ll.is_nan()
}

#[derive(Debug)]
struct MCState {
    g: Pcg64,
    now: u64,
    nodes: Vec<Box<dyn Any>>,
    deps: HashMap<u32, NodeSet>,
    inactive: HashMap<u32, u64>,
    delayed_updates: NodeSet,
    ll_total: LogLikelihood,
    sampleable: Vec<u32>,
    added: NodeSet,
    change_num: u32,
}
impl MCState {
    fn new(seed: u64) -> Self {
        Self {
            g: Pcg64::seed_from_u64(seed),
            now: 1,
            nodes: Vec::new(),
            deps: HashMap::new(),
            inactive: HashMap::new(),
            delayed_updates: Set64::new(),
            ll_total: 0.0,
            sampleable: Vec::new(),
            added: Set64::new(),
            change_num: 0,
        }
    }
    fn get_node<A: 'static>(&self, r: NodeRef<A>) -> &Node<A> {
        self.nodes[r.0 as usize].downcast_ref::<Node<A>>().unwrap()
    }
    fn get_node_mut<A: 'static>(&mut self, r: NodeRef<A>) -> &mut Node<A> {
        self.nodes[r.0 as usize].downcast_mut::<Node<A>>().unwrap()
    }
    fn modify_ll<A: 'static>(&mut self, r: NodeRef<A>, ll: LogLikelihood) {
        let this = self.get_node_mut(r);
        let old_ll = std::mem::replace(&mut this.ll, ll);
        self.ll_total += ll - old_ll;
    }
    fn modify_timestamp<A: 'static>(&mut self, r: NodeRef<A>, timestamp: u64, set: &mut NodeSet) {
        let this = self.get_node_mut(r);
        let old_timestamp = std::mem::replace(&mut this.timestamp, timestamp);
        if old_timestamp != timestamp {
            set.extend(self.deps.get(&r.0).unwrap().iter());
        }
    }
    fn add_node<A: 'static>(&mut self, n: Node<A>) -> NodeRef<A> {
        let a = self.nodes.len() as u32;
        if n.resampleable() {
            self.sampleable.push(a);
        }
        self.added.insert(a);
        self.ll_total += n.ll;
        self.nodes.push(Box::new(n));
        log::debug!("new node ref: {:?}", a);
        NodeRef::new(a)
    }
    fn add_dependency(&mut self, a: u32, b: u32) {
        self.deps.entry(a).or_default().insert(b);
    }
    fn propagate_updates<A: 'static>(&mut self, nodes: NodeSet) {
        let mut heap = BinaryHeap::from_iter(nodes.iter());
        while let Some(r) = heap.pop() {
            if self.inactive.contains_key(&r) {
                self.delayed_updates.insert(r);
                log::debug!("delayed update: {:?}", r);
                continue;
            }
            let r: NodeRef<A> = NodeRef::new(r); // crazy typing hack
            let n = self.get_node(r);
            if n.timestamp >= self.now {
                panic!(
                    "dependency violation: updating the already updated. node {:?} has timestamp ({}) >= now ({}). ",
                    r, n.timestamp, self.now
                );
            }
            if let Some(updater) = &n.updater {
                let mut set = Set64::new();
                let op = updater.update(r, self, &mut set);
                heap.extend(set.iter());
                match op {
                    NodeChange::ChangeValue(new_timestamp, value) => {
                        self.get_node_mut(r).value = value;
                        let mut set = Set64::new();
                        self.modify_timestamp(r, new_timestamp, &mut set);
                        heap.extend(set.iter());
                    }
                    NodeChange::ChangeDist(ll, dist) => {
                        self.get_node_mut(r).dist = dist;
                        self.modify_ll(r, ll);
                    }
                }
            }
        }
    }
    fn create_node<A: Clone + 'static>(
        &mut self,
        dist: DistType<A>,
        updater: Option<Box<dyn Updater<A>>>,
    ) -> NodeRef<A> {
        let (value, ll) = dist.sample(&mut self.g);
        let timestamp = self.now;
        self.add_node(Node {
            value,
            ll,
            dist,
            timestamp,
            updater,
        })
    }
    fn with_node_none<A: Clone + 'static>(&mut self, dist: DistType<A>) -> NodeRef<A> {
        self.create_node(dist, None)
    }
    fn with_node<A: Clone + 'static, B: Clone + 'static>(
        &mut self,
        dep: NodeRef<A>,
        f: fn(A) -> DistType<B>,
    ) -> NodeRef<B> {
        let dist = f(self.get_node(dep).value.clone());
        let nnew = self.create_node(dist, Some(Box::new(NodeDep1 { dep, f })));
        self.add_dependency(dep.0, nnew.0);
        nnew
    }
    fn with_node2<A: Clone + 'static, B: Clone + 'static, C: Clone + 'static>(
        &mut self,
        dep1: NodeRef<A>,
        dep2: NodeRef<B>,
        f: fn(A, B) -> DistType<C>,
    ) -> NodeRef<C> {
        let dist = f(
            self.get_node(dep1).value.clone(),
            self.get_node(dep2).value.clone(),
        );
        let nnew = self.create_node(dist, Some(Box::new(NodeDep2 { dep1, dep2, f })));
        self.add_dependency(dep1.0, nnew.0);
        self.add_dependency(dep2.0, nnew.0);
        nnew
    }
    // the algorithm
    fn mcmc<A: Clone + 'static>(&mut self, limit: u64, trace: NodeRef<A>) -> Vec<A> {
        todo!("ensure the initial state is accepted");
        let mut samples = Vec::new();
        for _ in 0..limit {
            samples.push(self.get_node(trace).value.clone());
            let accept = self.transition::<A>();
            if !accept {
                log::debug!("rejected");
                todo!("restore the state!");
            }
        }
        samples
    }
    fn transition<A: Clone + 'static>(&mut self) -> bool {
        self.now += 1;
        self.ll_total = 0.0;
        let changed_num_old = self.change_num;
        let sampleable_num = self.sampleable.len() as u32;
        let (changed, fwd_bwd) = self.resample_trace::<A>();
        self.propagate_updates::<A>(changed);
        let ll_diff = self.ll_total;
        let change_num = self.change_num;
        if rejected_ll(ll_diff) {
            return false;
        }
        let change_add = if change_num == changed_num_old {
            0.0
        } else {
            ((sampleable_num + changed_num_old) as f64).ln()
                - ((sampleable_num + change_num) as f64).ln()
        };
        let acceptance_ratio = ll_diff - fwd_bwd + change_add;
        acceptance_ratio >= 0.0 || self.g.random::<f64>() < acceptance_ratio.exp()
    }
    fn resample_trace<A: Clone + 'static>(&mut self) -> (NodeSet, LogLikelihood) {
        let r = loop {
            let i = self.g.random_range(0..self.sampleable.len());
            let r = self.sampleable[i];
            if !self.inactive.contains_key(&r) {
                break r;
            }
        };
        let mut g = std::mem::replace(&mut self.g, Pcg64::seed_from_u64(0));
        let r: NodeRef<A> = NodeRef::new(r);
        let n = self.get_node(r);
        let ((value, llv), fwd_bwd) = n.dist.resample(n.value.clone(), n.ll, &mut g);
        let _ = std::mem::replace(&mut self.g, g);
        // log::debug!("node {:?} resampled {:?}", r, n);
        let mut set = Set64::new();
        self.modify_ll(r, llv);
        self.modify_timestamp(r, self.now, &mut set);
        self.get_node_mut(r).value = value;
        (set, fwd_bwd)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn example1() {
        let mut state = MCState::new(1337);
        let c = state.with_node_none(uniform_s(0.0, 1.0));
        let d = state.with_node_none(bern_s(0.5));
        let e = state.with_node(c, bern_s);
        let result = state.with_node2(e, d, |e, d| dirac_s(e && d));
    }

    #[test]
    fn upp1() {
        let mut state = MCState::new(1337);
        let a = state.with_node_none(uniform_s(0.0, 1.0));
        let b = state.with_node(a, |a| dirac_s(a + 2.0));
        let c = state.with_node(a, |a| dirac_s(a + 3.0));
        let ab = state.with_node2(a, b, |a, b| dirac_s(a + b));
        let abc = state.with_node2(ab, c, |ab, c| uniform_s(ab, c));
    }
}
