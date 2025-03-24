use crate::distributions::{
    Bernoulli, Categorical, Dirac, Distribution, LogLikelihood, ReDistribution, Uniform, Uniformly,
};
use log::Log;
use rand::{Rng, SeedableRng, random_iter, random_range};
use rand_pcg::{Pcg64, rand_core::le};
use std::{
    any::Any, cmp::Reverse, collections::{BinaryHeap, HashMap}, fmt::Debug, marker::PhantomData
};
use tinyset::Set64;

/// A vector that supports simple transactions (i.e., batch undo).
struct VecUndo<A> {
    vec: Vec<A>,
    changes: HashMap<usize, A>,
}
impl<A> VecUndo<A> {
    fn new() -> Self {
        Self {
            vec: Vec::new(),
            changes: HashMap::new(),
        }
    }
    fn get(&self, i: usize) -> &A {
        self.changes.get(&i).unwrap_or(&self.vec[i])
    }
    fn set(&mut self, i: usize, a: A) {
        self.changes.insert(i, a);
    }
    fn push_forced(&mut self, a: A) {
        self.vec.push(a)
    }
    fn commit(&mut self) {
        for (i, a) in self.changes.drain() {
            self.vec[i] = a;
        }
    }
    fn rollback(&mut self) {
        self.changes.clear();
    }
    fn len(&self) -> usize {
        self.vec.len()
    }
}

/// A hash table that supports simple transactions (i.e., batch undo).
struct HashMapUndo<K, V> {
    map: HashMap<K, V>,
    changes: HashMap<K, V>,
}
impl<K: Eq + std::hash::Hash, V> HashMapUndo<K, V> {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            changes: HashMap::new(),
        }
    }
    fn contains_key(&self, k: &K) -> bool {
        self.changes.contains_key(k) || self.map.contains_key(k)
    }
    fn get(&self, k: &K) -> &V {
        self.changes.get(k).unwrap_or(&self.map[k])
    }
    fn set(&mut self, k: K, v: V) {
        self.changes.insert(k, v);
    }
    fn set_forced(&mut self, k: K, v: V) {
        self.map.insert(k, v);
    }
    fn commit(&mut self) {
        for (k, v) in self.changes.drain() {
            self.map.insert(k, v);
        }
    }
    fn rollback(&mut self) {
        self.changes.clear();
    }
}

/// Distributions as stored in nodes.
/// can sample and resample.
/// Observed distributions have fixed values.
#[derive(Debug)]
pub enum DistType<A> {
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
}
/// Distribution with its type erased.
/// 
/// `resample` has a dynamically typed parameter, but DistType knows its type.
/// `SomeDist` hides the type of `old_value` from the caller, so that
/// multiple types of distributions can be stored in the same container.
/// 
trait SomeDist {
    fn resample(
        &self,
        old_value: &dyn Any,
        old_ll: LogLikelihood,
        g: &mut Pcg64,
    ) -> ((Box<dyn Any>, LogLikelihood), LogLikelihood);
}
impl<A: Clone + 'static> SomeDist for DistType<A> {
    fn resample(
        &self,
        old_value: &dyn Any,
        old_ll: LogLikelihood,
        g: &mut Pcg64,
    ) -> ((Box<dyn Any>, LogLikelihood), LogLikelihood) {
        let old_value: &A = old_value.downcast_ref().unwrap();
        let ((new_value, ll), fwd_bwd) = match self {
            DistType::Dirac(_) => unreachable!(),
            DistType::Resampleable(dist) => dist.resample(old_value.clone(), old_ll, g),
            DistType::Observed(_, dist) => dist.resample(old_value.clone(), old_ll, g),
        };
        ((Box::new(new_value), ll), fwd_bwd)
    }
}
pub fn dirac_s<A: Clone>(a: A) -> DistType<A> {
    DistType::Dirac(a)
}
pub fn bern_s(p: f64) -> DistType<bool> {
    DistType::Resampleable(Box::new(Bernoulli::new(p)))
}
pub fn uniform_s(lower: f64, upper: f64) -> DistType<f64> {
    DistType::Resampleable(Box::new(Uniform::new(lower, upper)))
}
pub fn conditioned<A>(a: A, dist: DistType<A>) -> DistType<A> {
    match dist {
        DistType::Resampleable(d) => DistType::Observed(a, d),
        _ => panic!("conditioned on a non-resampleable distribution"),
    }
}

/// A node in the probabilistic program.
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
type RefSet = Set64<u32>;

/// mutable global state.
#[derive(Debug, Clone)]
pub struct Stats {
    now: u64,
    delayed_updates: RefSet,
    ll_total: LogLikelihood,
    change_num: u32,
}
impl Stats {
    pub fn new() -> Self {
        Self {
            now: 1,
            delayed_updates: Set64::new(),
            ll_total: 0.0,
            change_num: 0,
        }
    }
}

/// a node.
#[derive(Debug)]
struct Node<A> {
    ll: LogLikelihood,
    timestamp: u64,
    value: A,
    dist: DistType<A>,
    updater: Option<Box<dyn Updater>>,
}
impl<A> Node<A> {
    fn resampleable(&self) -> bool {
        match self.dist {
            DistType::Resampleable(_) => true,
            _ => false,
        }
    }
}

/// state in nodes. mutable during sampling.
pub struct Contents {
    lls: VecUndo<LogLikelihood>,
    timestamps: VecUndo<u64>,
    value: VecUndo<Box<dyn Any>>,
    dist: VecUndo<Box<dyn SomeDist>>,
    inactive: HashMapUndo<u32, u64>,
}
impl Contents {
    pub fn new() -> Self {
        Self {
            lls: VecUndo::new(),
            timestamps: VecUndo::new(),
            value: VecUndo::new(),
            dist: VecUndo::new(),
            inactive: HashMapUndo::new(),
        }
    }
    fn commit(&mut self) {
        self.lls.commit();
        self.timestamps.commit();
        self.value.commit();
        self.dist.commit();
        self.inactive.commit();
    }
    fn rollback(&mut self) {
        self.lls.rollback();
        self.timestamps.rollback();
        self.value.rollback();
        self.dist.rollback();
        self.inactive.rollback();
    }
    fn get_ll(&self, i: u32) -> LogLikelihood {
        *self.lls.get(i as usize)
    }
    fn set_ll(&mut self, i: u32, ll: LogLikelihood) {
        self.lls.set(i as usize, ll);
    }
    fn get_timestamp(&self, i: u32) -> u64 {
        *self.timestamps.get(i as usize)
    }
    fn set_timestamp(&mut self, i: u32, timestamp: u64) {
        self.timestamps.set(i as usize, timestamp);
    }
    fn get_value<A: 'static>(&self, r: NodeRef<A>) -> &A {
        self.value.get(r.0 as usize).downcast_ref::<A>().unwrap()
    }
    fn get_value_raw(&self, i: u32) -> &Box<dyn Any> {
        self.value.get(i as usize)
    }
    fn set_value<A: 'static>(&mut self, r: NodeRef<A>, v: A) {
        self.value.set(r.0 as usize, Box::new(v));
    }
    fn set_value_raw(&mut self, i: u32, v: Box<dyn Any>) {
        self.value.set(i as usize, v);
    }
    fn get_dist_raw(&self, i: u32) -> &Box<dyn SomeDist> {
        self.dist.get(i as usize)
    }
    fn set_dist<A: Clone + 'static>(&mut self, r: NodeRef<A>, d: DistType<A>) {
        self.dist.set(r.0 as usize, Box::new(d));
    }
}

/// The graph structure. immutable during sampling.
pub struct Graph {
    nodes: RefSet,
    deps: HashMap<u32, RefSet>,
    resampleable: Vec<u32>,
    updaters: Vec<Option<Box<dyn Updater>>>,
}
impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Set64::new(),
            deps: HashMap::new(),
            resampleable: Vec::new(),
            updaters: Vec::new(),
        }
    }
    fn add_dependency(&mut self, a: u32, b: u32) {
        self.deps.entry(a).or_default().insert(b);
    }
}

trait Updater {
    fn update(
        &self,
        updated: u32,
        stats: &mut Stats,
        contents: &mut Contents,
        graph: &Graph,
        set: &mut RefSet,
    );
}
impl Debug for dyn Updater {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Updater>")
    }
}

#[derive(Debug)]
struct NodeDep1<A, B> {
    dep: NodeRef<A>,
    f: fn(A) -> DistType<B>,
}
impl<A: Clone + 'static, B: Clone + 'static> Updater for NodeDep1<A, B> {
    fn update(
        &self,
        updated: u32,
        stats: &mut Stats,
        contents: &mut Contents,
        graph: &Graph,
        set: &mut RefSet,
    ) {
        assert!(
            !contents.inactive.contains_key(&self.dep.0),
            "activity violation: updating the inactive. node {:?} is inactive.",
            self.dep
        );
        assert!(
            !contents.inactive.contains_key(&updated),
            "activity violation: updating the inactive. node {:?} is inactive.",
            updated
        );
        let dep_timestamp = contents.get_timestamp(self.dep.0);
        let this_timestamp = contents.get_timestamp(updated);
        assert!(
            this_timestamp < dep_timestamp,
            "dependency violation: node {:?} depends on {:?}, but has timestamp ({}) >= ({})",
            updated,
            self.dep,
            this_timestamp,
            dep_timestamp
        );
        let updated: NodeRef<B> = NodeRef::new(updated);
        let dist = (self.f)(contents.get_value(self.dep).clone());
        match dist {
            DistType::Dirac(value) => {
                let now = stats.now;
                modify_timestamp(contents, graph, updated.0, now, set);
                contents.set_value(updated, value);
            }
            DistType::Resampleable(ref d) => {
                let this_value = contents.get_value(updated);
                modify_ll(stats, contents, updated.0, d.log_density(this_value));
                contents.set_dist(updated, dist);
            }
            DistType::Observed(_, ref d) => {
                let this_value = contents.get_value(updated);
                modify_ll(stats, contents, updated.0, d.log_density(this_value));
                contents.set_dist(updated, dist);
            }
        }
    }
}

#[derive(Debug)]
struct NodeDep2<A, B, C> {
    dep1: NodeRef<A>,
    dep2: NodeRef<B>,
    f: fn(A, B) -> DistType<C>,
}
impl<A: Clone + 'static, B: Clone + 'static, C: Clone + 'static> Updater for NodeDep2<A, B, C> {
    fn update(
        &self,
        updated: u32,
        stats: &mut Stats,
        contents: &mut Contents,
        graph: &Graph,
        set: &mut RefSet,
    ) {
        assert!(
            !contents.inactive.contains_key(&self.dep1.0),
            "activity violation: updating the inactive. node {:?} is inactive.",
            self.dep1
        );
        assert!(
            !contents.inactive.contains_key(&self.dep2.0),
            "activity violation: updating the inactive. node {:?} is inactive.",
            self.dep2
        );
        assert!(
            !contents.inactive.contains_key(&updated),
            "activity violation: updating the inactive. node {:?} is inactive.",
            updated
        );
        let dep1_timestamp = contents.get_timestamp(self.dep1.0);
        let dep2_timestamp = contents.get_timestamp(self.dep2.0);
        let this_timestamp = contents.get_timestamp(updated);
        assert!(
            this_timestamp < dep1_timestamp || this_timestamp < dep2_timestamp,
            "dependency violation: node {:?} depends on {:?} and {:?}, but has timestamp ({}) >= ({}) and ({})",
            updated,
            self.dep1,
            self.dep2,
            this_timestamp,
            dep1_timestamp,
            dep2_timestamp
        );
        let updated: NodeRef<C> = NodeRef::new(updated);
        let dist = (self.f)(
            contents.get_value(self.dep1).clone(),
            contents.get_value(self.dep2).clone(),
        );
        match dist {
            DistType::Dirac(value) => {
                let now = stats.now;
                modify_timestamp(contents, graph, updated.0, now, set);
                contents.set_value(updated, value);
            }
            DistType::Resampleable(ref d) => {
                let this_value = contents.get_value(updated);
                modify_ll(stats, contents, updated.0, d.log_density(this_value));
                contents.set_dist(updated, dist);
            }
            DistType::Observed(_, ref d) => {
                let this_value = contents.get_value(updated);
                modify_ll(stats, contents, updated.0, d.log_density(this_value));
                contents.set_dist(updated, dist);
            }
        }
    }
}

fn modify_ll(stats: &mut Stats, contents: &mut Contents, i: u32, ll: LogLikelihood) {
    let old_ll = contents.get_ll(i);
    stats.ll_total += ll - old_ll;
    contents.set_ll(i, ll);
}

fn modify_timestamp(
    contents: &mut Contents,
    graph: &Graph,
    i: u32,
    timestamp: u64,
    set: &mut RefSet,
) {
    let old_timestamp = contents.get_timestamp(i);
    if old_timestamp != timestamp {
        let deps = graph.deps.get(&i);
        if let Some(deps) = deps {
            set.extend(deps.iter());
        }
    }
    contents.set_timestamp(i, timestamp);
}

fn add_node<A: Clone + 'static>(
    stats: &mut Stats,
    contents: &mut Contents,
    graph: &mut Graph,
    node: Node<A>,
) -> NodeRef<A> {
    let i = contents.value.len() as u32;
    let resampleable = node.resampleable();

    contents.lls.push_forced(node.ll);
    contents.timestamps.push_forced(node.timestamp);
    contents.value.push_forced(Box::new(node.value));
    contents.dist.push_forced(Box::new(node.dist));

    graph.nodes.insert(i);
    if resampleable {
        graph.resampleable.push(i);
    }
    graph.updaters.push(node.updater);

    stats.ll_total += node.ll;

    log::debug!("new node: {:?}", i);
    NodeRef::new(i)
}

fn propagate_updates(stats: &mut Stats, contents: &mut Contents, graph: &Graph, nodes: RefSet) {
    let mut heap = BinaryHeap::from_iter(nodes.iter().map(Reverse));
    let mut last = None;
    while let Some(Reverse(a)) = heap.pop() {
        // dedup
        if let Some(v) = last {
            if v == a {
                continue;
            }
        }
        last = Some(a);

        if contents.inactive.contains_key(&a) {
            stats.delayed_updates.insert(a);
            log::debug!("delayed update: {:?}", a);
            continue;
        }
        let n_timestamp = contents.get_timestamp(a);
        assert!(
            n_timestamp < stats.now,
            "dependency violation: updating the already updated. node {:?} has timestamp ({}) >= now ({}). ",
            a,
            n_timestamp,
            stats.now
        );
        if let Some(updater) = graph.updaters[a as usize].as_ref() {
            let mut set = Set64::new();
            updater.update(a, stats, contents, graph, &mut set);
            heap.extend(set.iter().map(Reverse));
        }
    }
}

fn create_node<A: Clone + 'static>(
    g: &mut Pcg64,
    stats: &mut Stats,
    contents: &mut Contents,
    graph: &mut Graph,
    dist: DistType<A>,
    updater: Option<Box<dyn Updater>>,
) -> NodeRef<A> {
    let (value, ll) = dist.sample(g);
    let timestamp = stats.now;
    add_node(
        stats,
        contents,
        graph,
        Node {
            ll,
            timestamp,
            value,
            dist,
            updater,
        },
    )
}

fn with_node_none<A: Clone + 'static>(
    g: &mut Pcg64,
    stats: &mut Stats,
    contents: &mut Contents,
    graph: &mut Graph,
    dist: DistType<A>,
) -> NodeRef<A> {
    create_node(g, stats, contents, graph, dist, None)
}

fn with_node<A: Clone + 'static, B: Clone + 'static>(
    g: &mut Pcg64,
    stats: &mut Stats,
    contents: &mut Contents,
    graph: &mut Graph,
    dep: NodeRef<A>,
    f: fn(A) -> DistType<B>,
) -> NodeRef<B> {
    let dist = f(contents.get_value(dep).clone());
    let updater = Box::new(NodeDep1 { dep, f });
    let r = create_node(g, stats, contents, graph, dist, Some(updater));
    graph.add_dependency(dep.0, r.0);
    r
}

fn with_node2<A: Clone + 'static, B: Clone + 'static, C: Clone + 'static>(
    g: &mut Pcg64,
    stats: &mut Stats,
    contents: &mut Contents,
    graph: &mut Graph,
    dep1: NodeRef<A>,
    dep2: NodeRef<B>,
    f: fn(A, B) -> DistType<C>,
) -> NodeRef<C> {
    let dist = f(
        contents.get_value(dep1).clone(),
        contents.get_value(dep2).clone(),
    );
    let updater = Box::new(NodeDep2 { dep1, dep2, f });
    let r = create_node(g, stats, contents, graph, dist, Some(updater));
    graph.add_dependency(dep1.0, r.0);
    graph.add_dependency(dep2.0, r.0);
    r
}

// the algorithm
fn rejected_ll(ll_diff: LogLikelihood) -> bool {
    ll_diff.is_nan() || ll_diff.is_infinite()
}
pub fn mcmc<A: Clone + 'static>(
    g: &mut Pcg64,
    stats: &mut Stats,
    contents: &mut Contents,
    graph: &Graph,
    limit: u64,
    trace: NodeRef<A>,
) -> Vec<A> {
    assert!(!rejected_ll(stats.ll_total));
    let mut samples = Vec::new();
    for _ in 0..limit {
        samples.push(contents.get_value(trace).clone());
        let old_stats = stats.clone();
        let accept = transition(g, stats, contents, graph);
        if accept {
            contents.commit();
        } else {
            log::debug!("rejected");
            contents.rollback();
            *stats = old_stats;
        }
    }
    samples
}
fn transition(g: &mut Pcg64, stats: &mut Stats, contents: &mut Contents, graph: &Graph) -> bool {
    stats.now += 1;
    stats.ll_total = 0.0;
    let changed_num_old = stats.change_num;
    let (changed, fwd_bwd) = resample_trace(g, stats, contents, graph);
    propagate_updates(stats, contents, graph, changed);
    let ll_diff = stats.ll_total;
    let change_num = stats.change_num;
    if rejected_ll(ll_diff) {
        return false;
    }
    let resampleable_num = graph.resampleable.len() as u32;
    let change_add = if change_num == changed_num_old {
        0.0
    } else {
        ((resampleable_num + changed_num_old) as f64).ln()
            - ((resampleable_num + change_num) as f64).ln()
    };
    let acceptance_ratio = ll_diff - fwd_bwd + change_add;
    acceptance_ratio >= 0.0 || g.random::<f64>() < acceptance_ratio.exp()
}
fn resample_trace(
    g: &mut Pcg64,
    stats: &mut Stats,
    contents: &mut Contents,
    graph: &Graph,
) -> (RefSet, LogLikelihood) {
    let a = loop {
        let i = g.random_range(0..graph.resampleable.len());
        let r = graph.resampleable[i];
        if !contents.inactive.contains_key(&r) {
            break r;
        }
    };
    let value = contents.get_value_raw(a);
    let dist = contents.get_dist_raw(a);
    let ((value, llv), fwd_bwd) = dist.resample(value.as_ref(), contents.get_ll(a), g);
    let mut set = Set64::new();
    modify_ll(stats, contents, a, llv);
    modify_timestamp(contents, graph, a, stats.now, &mut set);
    contents.set_value_raw(a, value);
    (set, fwd_bwd)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn example1() {
        let mut g = Pcg64::seed_from_u64(1337);
        let mut stats = Stats::new();
        let mut contents = Contents::new();
        let mut graph = Graph::new();
        let c = with_node_none(
            &mut g,
            &mut stats,
            &mut contents,
            &mut graph,
            uniform_s(0.0, 1.0),
        );
        let d = with_node_none(&mut g, &mut stats, &mut contents, &mut graph, bern_s(0.5));
        let e = with_node(&mut g, &mut stats, &mut contents, &mut graph, c, bern_s);
        let result = with_node2(
            &mut g,
            &mut stats,
            &mut contents,
            &mut graph,
            e,
            d,
            |e, d| dirac_s(e && d),
        );
        let trace = mcmc(&mut g, &mut stats, &mut contents, &graph, 20, result);
        print!("{:?}", trace);
    }

    #[test]
    fn upp1() {
        let mut g = Pcg64::seed_from_u64(1337);
        let mut stats = Stats::new();
        let mut contents = Contents::new();
        let mut graph = Graph::new();

        let a = with_node_none(
            &mut g,
            &mut stats,
            &mut contents,
            &mut graph,
            uniform_s(0.0, 1.0),
        );
        let b = with_node(&mut g, &mut stats, &mut contents, &mut graph, a, |a| {
            dirac_s(a + 2.0)
        });
        let c = with_node(&mut g, &mut stats, &mut contents, &mut graph, a, |a| {
            dirac_s(a + 3.0)
        });
        let ab = with_node2(
            &mut g,
            &mut stats,
            &mut contents,
            &mut graph,
            a,
            b,
            |a, b| dirac_s(a + b),
        );
        let abc = with_node2(
            &mut g,
            &mut stats,
            &mut contents,
            &mut graph,
            ab,
            c,
            |ab, c| uniform_s(ab, c),
        );
        let trace = mcmc(&mut g, &mut stats, &mut contents, &graph, 20, abc);
        print!("{:?}", trace);
    }
}
