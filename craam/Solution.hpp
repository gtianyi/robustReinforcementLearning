// This file is part of CRAAM, a C++ library for solving plain
// and robust Markov decision processes.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "craam/Transition.hpp"
#include "craam/definitions.hpp"

#include <cmath>

namespace craam {

/**
 * A set of values that represent a solution to a plain MDP.
 *
 * @tparam PolicyType Type of the policy used (int deterministic, numvec
 * stochastic, but could also have multiple components (such as an action and
 * transition probability) )
 */
template <class PolicyType> struct Solution {
    /// Value function
    numvec valuefunction;
    /// Policy of the decision maker (and nature if applicable) for each state
    vector<PolicyType> policy;
    /// Bellman residual of the computation
    prec_t residual;
    /// Number of iterations taken
    long iterations;
    /// Time taken to solve the problem
    prec_t time;

    Solution()
        : valuefunction(0), policy(0), residual(-1), iterations(-1), time(std::nan("")) {}

    /// Empty solution for a problem with statecount states
    Solution(size_t statecount)
        : valuefunction(statecount, 0.0), policy(statecount), residual(-1),
          iterations(-1), time(nan("")) {}

    /// Empty solution for a problem with a given value function and policy
    Solution(numvec valuefunction, vector<PolicyType> policy, prec_t residual = -1,
             long iterations = -1, double time = nan(""))
        : valuefunction(move(valuefunction)), policy(move(policy)), residual(residual),
          iterations(iterations), time(time) {}

    /**
  Computes the total return of the solution given the initial
  distribution.
  @param initial The initial distribution
   */
    prec_t total_return(const Transition& initial) const {
        if (initial.max_index() >= (long)valuefunction.size())
            throw invalid_argument("Too many indexes in the initial distribution.");
        return initial.value(valuefunction);
    };
};

/// A solution with a deterministic policy
using DetermSolution = Solution<long>;

/// Solution to an S,A rectangular robust problem to an MDP
using SARobustSolution = Solution<pair<long, numvec>>;

/// Solution to an S-rectangular robust problem to an MDP
using SRobustSolution = Solution<pair<numvec, vector<numvec>>>;

} // namespace craam
