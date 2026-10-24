/* Standalone scheduler verification; not linked into the DA library.
 * Counts describe adBinomial's schedule, not instrumentation of model calls. */
#include "adBinomial.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  uint64_t forward_steps,reverse_steps,peak_snapshots,snapshot_pushes,snapshot_restores;
} Schedule;

static int count_schedule(int nt,int budget,Schedule *s) {
  int action,step,live=0;
  *s=(Schedule){0};
  adBinomial_init(nt,budget,0);
  while(adBinomial_next(&action,&step)) {
    switch(action) {
      case PUSHSNAP:
        ++s->snapshot_pushes; ++live;
        if(live>(int)s->peak_snapshots) s->peak_snapshots=(uint64_t)live;
        break;
      case POPSNAP: --live; ++s->snapshot_restores; break;
      case LOOKSNAP: ++s->snapshot_restores; break;
      case ADVANCE: ++s->forward_steps; break;
      case FIRSTTURN: case TURN: ++s->forward_steps; ++s->reverse_steps; break;
      default: return 1;
    }
    if(live<0 || live>budget) return 2;
  }
  return live!=0 || s->reverse_steps!=(uint64_t)nt;
}

int main(int argc,char **argv) {
  Schedule s;
  if(argc==3) {
    int nt=atoi(argv[1]),budget=atoi(argv[2]);
    if(nt<1 || nt>1000000 || budget<1 || budget>90) return 1;
    if(count_schedule(nt,budget,&s)) return 2;
    printf("{\"forward_steps\":%llu,\"reverse_steps\":%llu,\"peak_snapshots\":%llu,"
           "\"snapshot_pushes\":%llu,\"snapshot_restores\":%llu}\n",
           (unsigned long long)s.forward_steps,(unsigned long long)s.reverse_steps,
           (unsigned long long)s.peak_snapshots,(unsigned long long)s.snapshot_pushes,
           (unsigned long long)s.snapshot_restores);
    return 0;
  }
  if(argc!=1) return 1;
  const int budgets[]={1,2,4,8,16,90};
  const int lengths[]={1,7,12,33,300};
  const uint64_t expected[]={45150,4900,1938,1280,1029};
  for(int t=0;t<5;++t) for(int b=0;b<6;++b) {
    if(count_schedule(lengths[t],budgets[b],&s)) return 2;
    if(t==4 && b<5 && s.forward_steps!=expected[b]) return 3;
  }
  puts("Native binomial scheduler: snapshot bounds, balanced saves/restores and reverse counts passed");
  return 0;
}
