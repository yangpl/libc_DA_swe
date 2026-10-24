/* Verify the scheduler against the closed-form binomial work bound and the
 * original block adjoint. No external checkpointing example is required.
 * Include production C to exercise placement at large limits. */
#include "../swe2d_4dvar.c"
#include <assert.h>
#include <stdio.h>

int block_swe_evaluate(const double *,int,int,double,double,double,int,int,int,
                       const int *,const double *,double,double *,double *,double *);

/* Independent combinatorial work oracle, without replay/placement logic.
 * Let r be the smallest level with C(c+r,r) >= ns. The state-only forward
 * schedule takes r*ns - C(c+r,r-1) steps; SWE adds its final endpoint step. */
static uint64_t choose(int n,int k) {
  if(k<0 || k>n) return 0;
  if(k>n-k) k=n-k;
  uint64_t value=1;
  for(int j=1;j<=k;++j) value=value*(uint64_t)(n-k+j)/(uint64_t)j;
  return value;
}

static uint64_t expected_forward_steps(int ns,int slots) {
  int r=0;
  while(choose(slots+r,r)<(uint64_t)ns) ++r;
  return (uint64_t)r*(uint64_t)ns-choose(slots+r,r-1)+1;
}

static void check_case(int ns,int stride,int budget) {
  const int n=6,nd=18,m=4;
  int indices[4]={0,5,2,0}; /* corners and repeated sensors */
  double initial[18],gradient[18],reference[18],diag[2],refdiag[2],cost,refcost;
  double *obs=malloc((size_t)(ns/stride)*m*sizeof(double));
  assert(obs);
  for(int k=0;k<nd;++k) initial[k]=(k<n ? 1. : .02)+.01*sin(k*.73);
  for(int k=0;k<(ns/stride)*m;++k) obs[k]=1.+.003*cos(k*.51);
  SweCheckpointStats stats;
  assert(!block_swe_evaluate(initial,3,2,1.3,.8,.001,ns,stride,m,indices,obs,.02,
                             &refcost,reference,refdiag));
  assert(!swe_evaluate_binomial(initial,3,2,1.3,.8,.001,ns,stride,m,indices,obs,.02,
                                budget,&cost,gradient,diag,&stats));
  assert(cost==refcost);
  for(int k=0;k<nd;++k) assert(fabs(gradient[k]-reference[k])<=1e-12*fmax(1.,fabs(reference[k])));
  assert(diag[0]==refdiag[0] && diag[1]==refdiag[1]);
  int nc=budget<ns ? budget : (ns>1 ? ns-1 : 1);
  uint64_t predicted=expected_forward_steps(ns,nc);
  if(stats.forward_steps!=predicted) {
    fprintf(stderr,"Schedule mismatch: ns=%d, budget=%d: %llu != %llu\n",ns,budget,
            (unsigned long long)stats.forward_steps,(unsigned long long)predicted);
    abort();
  }
  assert(stats.checkpoint_slots==(uint64_t)nc);
  assert(stats.state_storage_bytes==(uint64_t)(nc+6)*nd*sizeof(double));
  assert(stats.recomputed_steps==stats.forward_steps-(uint64_t)ns);
  assert(stats.reverse_steps==(uint64_t)ns);
  assert(stats.stage_recomputations==(uint64_t)(ns-1));
  assert(stats.checkpoint_restores==(uint64_t)(ns-1));
  assert(!swe_evaluate_binomial(initial,3,2,1.3,.8,.001,ns,stride,m,indices,obs,.02,
                                budget,&cost,NULL,diag,&stats));
  assert(cost==refcost && stats.forward_steps==(uint64_t)ns);
  assert(!stats.checkpoint_slots && !stats.recomputed_steps && !stats.reverse_steps);
  assert(stats.state_storage_bytes==4*nd*sizeof(double));
  free(obs);
}

int main(void) {
  int actual[90];
  for(int ns=2;ns<=200;++ns) for(int nc=1;nc<ns && nc<=20;++nc) {
    cp_binomial_redistribute(actual,ns,nc);
    assert(actual[0]==0);
    for(int k=1;k<nc;++k) assert(actual[k]>actual[k-1] && actual[k]<ns-1);
    /* Enough slots for every useful input state: no replay is needed. */
    if(nc==ns-1) for(int k=0;k<nc;++k) assert(actual[k]==k);
  }
  assert(expected_forward_steps(1,1)==1);
  assert(expected_forward_steps(300,1)==44851);
  assert(expected_forward_steps(300,8)==981);
  assert(expected_forward_steps(300,32)==567);
  /* Exercise 64-bit placement at supported limits, including large c and
   * single-slot schedules. No million-step physical integration is needed. */
  for(int ns=999991;ns<=1000000;++ns) for(int nc=1;nc<=90;++nc) {
    cp_binomial_redistribute(actual,ns,nc);
    assert(actual[0]==0);
    for(int k=1;k<nc;++k) assert(actual[k]>actual[k-1] && actual[k]<ns-1);
  }
  for(int ns=1;ns<=80;++ns) for(int budget=1;budget<=ns+1 && budget<=16;++budget) {
    check_case(ns,1,budget);
    if(ns>1) check_case(ns,ns,budget);
  }
  const int budgets[]={1,2,4,8,16,32,90};
  for(size_t k=0;k<sizeof(budgets)/sizeof(*budgets);++k) {
    check_case(300,20,budgets[k]);
    check_case(511,7,budgets[k]);
    check_case(1000,25,budgets[k]);
  }
  double initial[12]={1,1,1,1},obs[1]={1},cost,diag[2],grad[12];
  int idx=0;
  for(int budget=0;budget<=91;budget+=91)
    assert(swe_evaluate_binomial(initial,2,2,1,1,.001,1,1,1,&idx,obs,.01,
                                 budget,&cost,grad,diag,NULL)==1);
  assert(swe_evaluate_binomial(initial,2,2,1,1,.001,SWE_MAX_STEPS+1,1,1,&idx,
                               obs,.01,8,&cost,grad,diag,NULL)==1);
  puts("Binomial work bounds, placement invariants, block-gradient equivalence, "
       "single-step/surplus budgets, streaming costs, and storage bounds passed");
  return 0;
}
