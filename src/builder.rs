use crate::metropolis::{
    Contents, DepLeaf, DepPair, DepTree, Dist, Graph, NodeID, NodeRef, Stats, bern_s, dirac_s,
    mcmc_step, uniform_s, with_node_none, with_node_tree,
};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::collections::HashMap;
use std::ops::BitAnd;
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};
fn gen_id() -> u32 {
    static COUNTER: AtomicU32 = AtomicU32::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

trait Builder<A> {
    fn build(
        &self,
        built: &mut HashMap<LeafID, NodeID>,
        g: &mut Pcg64,
        stats: &mut Stats,
        contents: &mut Contents,
        graph: &mut Graph,
    ) -> NodeRef<A>;
}

struct Val<A>(A);
impl<A: Clone + 'static> Val<A> {
    fn new(a: A) -> Self {
        Self(a)
    }
}
impl<A: Clone + 'static> Builder<A> for Val<Dist<A>> {
    fn build(
        &self,
        built: &mut HashMap<LeafID, NodeID>,
        g: &mut Pcg64,
        stats: &mut Stats,
        contents: &mut Contents,
        graph: &mut Graph,
    ) -> NodeRef<A> {
        with_node_none(g, stats, contents, graph, self.0.clone())
    }
}

struct Comp<A, B>(Rc<dyn Tree<A>>, fn(A) -> B);
impl<A: Clone + 'static, B: Clone + 'static> Comp<A, B> {
    fn new(tree: Rc<dyn Tree<A>>, f: fn(A) -> B) -> Self {
        Self(tree, f)
    }
}
impl<A: Clone + 'static, B: Clone + 'static> Builder<B> for Comp<A, Dist<B>> {
    fn build(
        &self,
        built: &mut HashMap<LeafID, NodeID>,
        g: &mut Pcg64,
        stats: &mut Stats,
        contents: &mut Contents,
        graph: &mut Graph,
    ) -> NodeRef<B> {
        let tree = self.0.to_deptree(built, g, stats, contents, graph);
        with_node_tree(g, stats, contents, graph, tree, Box::new(self.1.clone()))
    }
}

type LeafID = u32;
trait Tree<A: Clone> {
    fn to_deptree(
        &self,
        built: &mut HashMap<LeafID, NodeID>,
        g: &mut Pcg64,
        stats: &mut Stats,
        contents: &mut Contents,
        graph: &mut Graph,
    ) -> Box<dyn DepTree<A>>;
}
#[derive(Clone)]
struct Leaf<A> {
    id: LeafID,
    dist: Rc<dyn Builder<A>>,
}
impl<A: Clone + 'static> Leaf<A> {
    fn new(dist: Rc<dyn Builder<A>>) -> Self {
        Self { id: gen_id(), dist }
    }
    fn build(
        &self,
        built: &mut HashMap<LeafID, NodeID>,
        g: &mut Pcg64,
        stats: &mut Stats,
        contents: &mut Contents,
        graph: &mut Graph,
    ) -> NodeRef<A> {
        if let Some(&node) = built.get(&self.id) {
            return NodeRef::new(node);
        }
        let node = self.dist.build(built, g, stats, contents, graph);
        built.insert(self.id, node.0);
        node
    }
}
impl<A: Clone + 'static> Tree<A> for Leaf<A> {
    fn to_deptree(
        &self,
        built: &mut HashMap<LeafID, NodeID>,
        g: &mut Pcg64,
        stats: &mut Stats,
        contents: &mut Contents,
        graph: &mut Graph,
    ) -> Box<dyn DepTree<A>> {
        if built.contains_key(&self.id) {
            return Box::new(DepLeaf::new(NodeRef::new(built[&self.id])));
        }
        let node = self.build(built, g, stats, contents, graph);
        built.insert(self.id, node.0);
        Box::new(DepLeaf::new(node))
    }
}
#[derive(Clone)]
struct Pair<A, B>(Rc<dyn Tree<A>>, Rc<dyn Tree<B>>);
impl<A: Clone + 'static, B: Clone + 'static> Pair<A, B> {
    fn new(left: Rc<dyn Tree<A>>, right: Rc<dyn Tree<B>>) -> Self {
        Self(left, right)
    }
}
impl<A: Clone + 'static, B: Clone + 'static> Tree<(A, B)> for Pair<A, B> {
    fn to_deptree(
        &self,
        built: &mut HashMap<LeafID, NodeID>,
        g: &mut Pcg64,
        stats: &mut Stats,
        contents: &mut Contents,
        graph: &mut Graph,
    ) -> Box<dyn DepTree<(A, B)>> {
        let left = self.0.to_deptree(built, g, stats, contents, graph);
        let right = self.1.to_deptree(built, g, stats, contents, graph);
        Box::new(DepPair::new(left, right))
    }
}

struct Model<A> {
    result: NodeRef<A>,
    g: Pcg64,
    stats: Stats,
    contents: Contents,
    graph: Graph,
}
impl<A: Clone + 'static> Model<A> {
    fn new(node: Leaf<A>, seed: u64) -> Self {
        let mut g = Pcg64::seed_from_u64(seed);
        let mut stats = Stats::new();
        let mut contents = Contents::new();
        let mut graph = Graph::new();
        let mut built = HashMap::new();
        let result = node.build(&mut built, &mut g, &mut stats, &mut contents, &mut graph);
        Self {
            result,
            g,
            stats,
            contents,
            graph,
        }
    }
}
impl<A: Clone + 'static> Iterator for Model<A> {
    type Item = A;
    fn next(&mut self) -> Option<Self::Item> {
        let value = mcmc_step(
            &mut self.g,
            &mut self.stats,
            &mut self.contents,
            &self.graph,
            self.result,
        );
        Some(value)
    }
}

fn val<A: Clone + 'static>(a: A) -> Leaf<A> {
    Leaf::new(Rc::new(Val::new(dirac_s(a))))
}
fn flip(p: Leaf<f64>) -> Leaf<bool> {
    Leaf::new(Rc::new(Comp::new(Rc::new(p), bern_s)))
}
fn uniform(a: Leaf<f64>, b: Leaf<f64>) -> Leaf<f64> {
    Leaf::new(Rc::new(Comp::new(
        Rc::new(Pair::new(Rc::new(a), Rc::new(b))),
        |(lower, upper)| uniform_s(lower, upper),
    )))
}
impl BitAnd for Leaf<bool> {
    type Output = Leaf<bool>;
    fn bitand(self, rhs: Self) -> Self::Output {
        let pair = Pair::new(Rc::new(self), Rc::new(rhs));
        Leaf::new(Rc::new(Comp::new(
            Rc::new(pair),
            |(a, b)| dirac_s(a & b), // inefficient.
        )))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test() {
        let c = uniform(val(0.0), val(1.0));
        let d = flip(val(0.5));
        let e = flip(c);
        let result = d & e;
        let model = Model::new(result, 1337);
        let samples = model
            .skip(2000)
            .take(2000)
            .map(|a| if a { 1 } else { 0 })
            .sum::<i32>();
        let delta = (samples - 500).abs() as f64 / 2000.0;
        assert!(delta < 0.05, "delta: {}, samples: {}", delta, samples);
    }
}
