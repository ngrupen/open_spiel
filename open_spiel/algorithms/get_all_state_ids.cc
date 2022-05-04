// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/algorithms/get_all_state_ids.h"

namespace open_spiel {
namespace algorithms {
namespace {

// Walk a subgame and return all states contained in the subgames. This does
// a recursive tree walk, therefore all valid sequences must have finite number
// of actions. The state collection is key-indexed by the state's string
// representation so that duplicates are not added.
// Requires State::Clone() to be implemented.
// Use with extreme caution!
// Currently not implemented for simultaneous games.
void GetSubgameStateIDs(State* state,
                      std::map<std::string, std::unique_ptr<State>>* all_state_ids,
                      int depth_limit, int depth, bool include_terminals,
                      bool include_chance_states,
                      bool stop_at_duplicates) {
  if (state->IsTerminal()) {
    if (include_terminals) {
      // Include if not already present and then terminate recursion.
      std::string key = state->GetIDString();
      if (all_state_ids->find(key) == all_state_ids->end()) {
        (*all_state_ids)[key] = state->Clone();
      }
    }
    return;
  }

  if (depth_limit >= 0 && depth > depth_limit) {
    return;
  }

  if (!state->IsChanceNode() || include_chance_states) {
    // Decision node; add only if not already present
    std::string key = state->GetIDString();
    if (all_state_ids->find(key) == all_state_ids->end()) {
      (*all_state_ids)[key] = state->Clone();
    } else {
      // Duplicate node.
      if (stop_at_duplicates) {
        // Terminate, do not explore the same node twice
        return;
      }
    }
  }

  for (auto action : state->LegalActions()) {
    auto next_state = state->Clone();
    next_state->ApplyAction(action);
    GetSubgameStateIDs(next_state.get(), all_state_ids, depth_limit, depth + 1,
                     include_terminals, include_chance_states,
                     stop_at_duplicates);
  }
}

}  // namespace

std::map<std::string, std::unique_ptr<State>> GetAllStateIDs(
    const Game& game, int depth_limit, bool include_terminals,
    bool include_chance_states, bool stop_at_duplicates) {
  // Get the root state.
  std::unique_ptr<State> state = game.NewInitialState();
  std::map<std::string, std::unique_ptr<State>> all_state_ids;

  // Then, do a recursive tree walk to fill up the map.
  GetSubgameStateIDs(state.get(), &all_state_ids, depth_limit, 0, include_terminals,
                   include_chance_states, stop_at_duplicates);

  if (all_state_ids.empty()) {
    SpielFatalError("GetSubgameStateIDs returned 0 state IDs!");
  }

  return all_state_ids;
}

}  // namespace algorithms
}  // namespace open_spiel