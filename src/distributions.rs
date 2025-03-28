use std::fmt::Debug;

use rand::Rng;
use rand_pcg::Pcg64;

pub type LogLikelihood = f64;

pub trait Sampleable<A> {
    fn log_density(&self, x: &A) -> LogLikelihood;
    fn sample(&self, g: &mut Pcg64) -> (A, LogLikelihood);
}
impl<A> Debug for dyn Sampleable<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Distribution<{:?}>", std::any::type_name::<A>())
    }
}

pub trait Resampleable<A>: Sampleable<A> {
    fn resample(
        &self,
        old_value: A,
        old_ll: LogLikelihood,
        g: &mut Pcg64,
    ) -> ((A, LogLikelihood), LogLikelihood) {
        let s @ (_, ll) = self.sample(g);
        (s, ll - old_ll)
    }
    fn resampleable(&self) -> bool {
        true
    }
}
impl<A> Debug for dyn Resampleable<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ReDistribution<{:?}>", std::any::type_name::<A>())
    }
}

#[derive(Clone, Debug)]
pub struct Dirac<A: PartialEq + Clone>(A);
impl<A: PartialEq + Clone> Dirac<A> {
    pub fn new(a: A) -> Self {
        Self(a)
    }
}
impl<A: PartialEq + Clone> Sampleable<A> for Dirac<A> {
    fn sample(&self, _: &mut Pcg64) -> (A, LogLikelihood) {
        (self.0.clone(), 0.0)
    }
    fn log_density(&self, x: &A) -> LogLikelihood {
        if *x == self.0 { 0.0 } else { f64::NEG_INFINITY }
    }
}

#[derive(Clone, Debug)]
pub struct Bernoulli {
    p: f64,
    ll1: LogLikelihood,
    ll0: LogLikelihood,
}
impl Bernoulli {
    pub fn new(p: f64) -> Self {
        let ll1 = p.ln();
        let ll0 = (1.0 - p).ln();
        Self { p, ll1, ll0 }
    }
}
impl Sampleable<bool> for Bernoulli {
    fn sample(&self, g: &mut Pcg64) -> (bool, LogLikelihood) {
        let x = g.random_bool(self.p);
        if x {
            (true, self.ll1)
        } else {
            (false, self.ll0)
        }
    }
    fn log_density(&self, x: &bool) -> LogLikelihood {
        if *x { self.ll1 } else { self.ll0 }
    }
}
impl Resampleable<bool> for Bernoulli {}

#[derive(Clone, Debug)]
pub struct Uniformly<A: Clone> {
    list: Vec<A>,
    ll: LogLikelihood,
}
impl<A: Clone> Uniformly<A> {
    pub fn new(list: Vec<A>) -> Self {
        let ll = -(list.len() as f64).ln();
        Self { list, ll }
    }
}
impl<A: Clone> Sampleable<A> for Uniformly<A> {
    fn sample(&self, g: &mut Pcg64) -> (A, LogLikelihood) {
        let i = g.random_range(0..self.list.len());
        (self.list[i].clone(), self.ll)
    }
    fn log_density(&self, _: &A) -> LogLikelihood {
        self.ll
    }
}
impl<A: Clone> Resampleable<A> for Uniformly<A> {}

#[derive(Clone, Debug)]
pub struct Categorical<A: Clone + PartialEq>(Vec<(A, f64)>);
impl<A: Clone + PartialEq> Categorical<A> {
    pub fn new(list: Vec<(A, f64)>) -> Self {
        let total = list.iter().map(|(_, p)| p).sum::<f64>();
        let norm_list = list.into_iter().map(|(a, p)| (a, p / total)).collect();
        Self(norm_list)
    }
}
impl<A: Clone + PartialEq> Sampleable<A> for Categorical<A> {
    fn sample(&self, g: &mut Pcg64) -> (A, LogLikelihood) {
        let mut r = g.random::<f64>();
        for (a, p) in &self.0 {
            r -= p;
            if r < 0.0 {
                return (a.clone(), p.ln());
            }
        }
        unreachable!()
    }
    fn log_density(&self, x: &A) -> LogLikelihood {
        let item = self.0.iter().find(|(a, _)| a == x);
        match item {
            Some((_, p)) => p.ln(),
            None => f64::NEG_INFINITY,
        }
    }
}
impl<A: Clone + PartialEq> Resampleable<A> for Categorical<A> {}

#[derive(Clone, Debug)]
pub struct Uniform {
    lower: f64,
    upper: f64,
    range_ll: LogLikelihood,
}
impl Uniform {
    pub fn new(lower: f64, upper: f64) -> Self {
        let range_ll = -(upper - lower).ln();
        Self {
            lower,
            upper,
            range_ll,
        }
    }
}
impl Sampleable<f64> for Uniform {
    fn sample(&self, g: &mut Pcg64) -> (f64, LogLikelihood) {
        let x = g.random_range(self.lower..self.upper);
        (x, self.range_ll)
    }
    fn log_density(&self, x: &f64) -> LogLikelihood {
        if *x >= self.lower && *x <= self.upper {
            self.range_ll
        } else {
            f64::NEG_INFINITY
        }
    }
}
impl Resampleable<f64> for Uniform {}
