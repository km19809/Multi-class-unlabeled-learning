import numpy as np
import math
from bitarray import bitarray
import heapq


def get_prior(data, labels):

    labels = bitarray(list(labels == 1))
    folds = np.random.randint(5, size=len(data))

    (c_estimate, c_its_estimates) = tice(data, labels, 5, folds, None, nbIterations=2, maxSplits=500, minT=10)

    alpha = 1.0
    if c_estimate > 0:
        pos = float(labels.count()) / c_estimate
        tot = len(data)
        alpha = max(0.0, min(1.0, pos / tot))

    #print("c:\t" + str(c_estimate))
    #print("alpha:\t" + str(alpha))

    return alpha


def pick_delta(T):
  return max(0.025, 1/(1+0.004*T))



def low_c(data, label, delta, minT,c=0.5):
  T = float(data.count())
  if T<minT:
    return 0.0
  L = float((data&label).count())
  clow = L/T - math.sqrt(c*(1-c)*(1-delta)/(delta*T))
  return clow


def max_bepp(k):
    def fun(counts):
        return max([(0 if T_P[0]==0 else float(T_P[1])/(T_P[0]+k)) for T_P in counts])
    return fun


def generate_folds(folds):
    for fold in range(max(folds)+1):
        tree_train = bitarray(list(folds==fold))
        estimate = ~tree_train
        yield (tree_train, estimate)


def tice(data, labels, k, folds, delta=None, nbIterations=2, maxSplits=500, useMostPromisingOnly=False, minT=10, ):
    c_its_ests = []
    c_estimate = 0.5


    for it in range(nbIterations):

        c_estimates = []


        global c_cur_best # global so that it can be used for optimizing queue.
        for (tree_train, estimate) in generate_folds(folds):
            c_cur_best = low_c(estimate, labels, 1.0, minT, c=c_estimate)
            cur_delta = delta if delta else pick_delta(estimate.count())


            if useMostPromisingOnly:

                c_tree_best=0.0
                most_promising = estimate
                for tree_subset, estimate_subset in subsetsThroughDT(data, tree_train, estimate, labels, splitCrit=max_bepp(k), minExamples=minT, maxSplits=maxSplits, c_prior=c_estimate, delta=cur_delta):
                    tree_est_here = low_c(tree_subset,labels,cur_delta, 1,c=c_estimate)
                    if tree_est_here > c_tree_best:
                        c_tree_best = tree_est_here
                        most_promising = estimate_subset

                c_estimates.append(max(c_cur_best, low_c(most_promising, labels, cur_delta,minT, c=c_estimate)))

            else:

                for tree_subset, estimate_subset in subsetsThroughDT(data, tree_train, estimate, labels, splitCrit=max_bepp(k), minExamples=minT, maxSplits=maxSplits, c_prior=c_estimate, delta=cur_delta):
                    est_here = low_c(estimate_subset,labels,cur_delta, minT,c=c_estimate)
                    c_cur_best=max(c_cur_best, est_here)
                c_estimates.append(c_cur_best)


        c_estimate = sum(c_estimates)/float(len(c_estimates))
        c_its_ests.append(c_estimates)

    return c_estimate, c_its_ests




def subsetsThroughDT(data, tree_train, estimate, labels,splitCrit=max_bepp(5), minExamples=10, maxSplits=500, c_prior=0.5, delta=0.0):
  # This learns a decision tree and updates the label frequency lower bound for every tried split.
  # It splits every variable into 4 pieces: [0,.25[ , [.25, .5[ , [.5,.75[ , [.75,1]
  # The input data is expected to have only binary or continues variables with values between 0 and 1. To achieve this, the multivalued variables should be binarized and the continuous variables should be normalized

  # Max: Return all the subsets encountered

  all_data=tree_train|estimate

  borders=[.25, .5, .75]

  def makeSubsets(a):
      subsets = []
      options=bitarray(all_data)
      for b in borders:
          X_cond = bitarray(list((data[:,a]<b)))&options
          options&=~X_cond
          subsets.append(X_cond)
      subsets.append(options)
      return subsets

  conditionSets = [ makeSubsets(a) for a in range(data.shape[1])]

  priorityq = []
  heapq.heappush(priorityq,(-low_c(tree_train, labels, delta, 0, c=c_prior),-(tree_train&labels).count(), tree_train, estimate, set(range(data.shape[1])), 0))
  yield (tree_train, estimate)

  n=0
  minimumLabeled = 1
  while n<maxSplits and len(priorityq)>0:
    n+=1
    (ppos, neg_lab_count, subset_train, subset_estimate, available, depth) = heapq.heappop(priorityq)
    lab_count= -neg_lab_count

    best_a=-1
    best_score=-1
    best_subsets_train=[]
    best_subsets_estimate=[]
    best_lab_counts=[]
    uselessAs=set()



    for a in available:

      subsets_train=[X_cond&subset_train for X_cond in conditionSets[a]]
      subsets_estimate=[X_cond&subset_train for X_cond in conditionSets[a]]
      estimate_lab_counts = [(subset&labels).count() for subset in subsets_estimate]
      if max(estimate_lab_counts) < minimumLabeled:
        uselessAs.add(a)
      else:


        score = splitCrit([(subsub.count(), (subsub&labels).count()) for subsub in subsets_train])
        if score>best_score:
            best_score=score
            best_a=a
            best_subsets_train=subsets_train
            best_subsets_estimate=subsets_estimate
            best_lab_counts = estimate_lab_counts


    fake_split = len([subset for subset in best_subsets_estimate if subset.count()>0])==1

    if best_score > 0 and not fake_split:
      newAvailable = available-set([best_a])-uselessAs
      for subsub_train,subsub_estimate in zip(best_subsets_train, best_subsets_estimate):
        yield (subsub_train,subsub_estimate)
      minimumLabeled = c_prior*(1-c_prior)*(1-delta)/(delta*(1-c_cur_best)**2)

      for (subsub_lab_count, subsub_train, subsub_estimate) in zip(best_lab_counts, best_subsets_train, best_subsets_estimate):
          if subsub_lab_count>minimumLabeled:
            total = subsub_train.count()
            if total>minExamples: #stop criterion: minimum size for splitting
              train_lab_count = (subsub_train&labels).count()
              if lab_count!=0 and lab_count!=total: #stop criterion: purity
                heapq.heappush(priorityq,(-low_c(subsub_train, labels, delta, 0, c=c_prior), -train_lab_count, subsub_train, subsub_estimate, newAvailable, depth+1))

