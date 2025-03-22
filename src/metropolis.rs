use crate::distributions::{
    Bernoulli, Categorical, Dirac, Distribution, LogLikelihood, ReDistribution, Uniform, Uniformly,
};
use rand::{Rng, SeedableRng};
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
    Resampleable(Box<dyn Distribution<A>>),
    Observed(A, Box<dyn Distribution<A>>),
}
impl<A: Clone> DistType<A> {
    fn sample(&self, g: &mut Pcg64) -> (A, LogLikelihood) {
        match self {
            DistType::Dirac(a) => (a.clone(), 0.0),
            DistType::Resampleable(dist) => dist.sample(g),
            DistType::Observed(a, dist) => (a.clone(), dist.log_density(a)),
        }
    }
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

// to update a node
trait Updater<A> {
    fn update(&self, updated: NodeRef<A>, state: &mut MCState, heap: &mut BinaryHeap<u32>);
}
impl<A> Debug for dyn Updater<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Updater<{:?}>", std::any::type_name::<A>())
    }
}
struct NodeDep1<A> {
    dep: NodeRef<A>,
    f: fn(A) -> DistType<A>,
}
impl<A> NodeDep1<A> {
    fn new(dep: NodeRef<A>, f: fn(A) -> DistType<A>) -> Self {
        Self { dep, f }
    }
}
impl<A: Clone + 'static> Updater<A> for NodeDep1<A> {
    fn update(&self, updated: NodeRef<A>, state: &mut MCState, _: &mut NodeQueue) {
        let now = state.now;
        if state.inactive.contains_key(&self.dep.0) {
            panic!(
                "activity violation: updating the inactive. node {:?} is inactive.",
                self.dep
            );
        }
        if state.inactive.contains_key(&updated.0) {
            panic!(
                "activity violation: updating the inactive. node {:?} is inactive.",
                updated
            );
        }
        let dep = state.get_node(self.dep);
        let dep_timestamp = dep.timestamp;
        let dep_value = dep.value.clone();
        let this = state.get_node_mut(updated);
        if dep_timestamp <= this.timestamp {
            panic!(
                "dependency violation: node {:?} depends on {:?}, but has timestamp ({}) >= ({})",
                updated, self.dep, this.timestamp, dep_timestamp
            );
        }
        let dist = (self.f)(dep_value);
        match dist {
            DistType::Dirac(value) => {
                this.value = value;
                this.timestamp = now;
            }
            DistType::Resampleable(ref d) => {
                this.ll = d.log_density(&this.value);
                this.dist = dist;
            }
            DistType::Observed(_, ref d) => {
                this.ll = d.log_density(&this.value);
                this.dist = dist;
            }
        }
        todo!("update timestamp and total log likelihood whan changing this")
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
    change_num: u64,
}
impl MCState {
    fn get_node<A: 'static>(&self, r: NodeRef<A>) -> &Node<A> {
        self.nodes[r.0 as usize].downcast_ref::<Node<A>>().unwrap()
    }
    fn get_node_mut<A: 'static>(&mut self, r: NodeRef<A>) -> &mut Node<A> {
        self.nodes[r.0 as usize].downcast_mut::<Node<A>>().unwrap()
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
    fn add_dependency<A: 'static>(&mut self, a: u32, b: u32) {
        self.deps.entry(a).or_default().insert(b);
    }
    fn modify_node<A: 'static>(&mut self, r: NodeRef<A>, n: Node<A>, heap: &mut NodeQueue) {
        let n_ll = n.ll;
        let n_timestamp = n.timestamp;
        let old = std::mem::replace(self.get_node_mut(r), n);
        self.ll_total += n_ll - old.ll;
        if n_timestamp != old.timestamp {
            let deps = self.deps.get(&r.0);
            if let Some(deps) = deps {
                heap.extend(deps.iter());
            }
        }
    }
    fn propagate_updates<A: Debug + 'static>(&mut self, nodes: NodeSet) {
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
                let new_node = updater.update(r, self, &mut heap);
                self.modify_node(r, new_node, &mut heap);
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
    fn with_node<A: Clone + 'static>(
        &mut self,
        r: NodeRef<A>,
        f: fn(A) -> DistType<A>,
    ) -> NodeRef<A> {
        let dist = f(self.get_node(r).value.clone());
        let nnew = self.create_node(dist, Some(Box::new(NodeDep1::new(r, f))));
        self.add_dependency::<A>(r.0, nnew.0);
        nnew
    }
}

#[cfg(test)]
mod test {
    pub fn test() {
        println!("Hello, world!");
    }
}
