// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/tic_tac_toe.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace tic_tac_toe {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"tic_tac_toe",
    /*long_name=*/"Tic Tac Toe",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TicTacToeGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kCross;
    case 1:
      return CellState::kNought;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
      return CellState::kEmpty;
  }
}

std::string StateToString(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return ".";
    case CellState::kNought:
      return "o";
    case CellState::kCross:
      return "x";
    default:
      SpielFatalError("Unknown state.");
  }
}

CellState InvertCellState(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return CellState::kEmpty;
    case CellState::kNought:
      return CellState::kCross;
    case CellState::kCross:
      return CellState::kNought;
    default:
      SpielFatalError("Unknown state.");
  }
}

void TicTacToeState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(board_[move], CellState::kEmpty);
  board_[move] = PlayerToState(CurrentPlayer());
  if (HasLine(current_player_)) {
    outcome_ = current_player_;
  }
  current_player_ = 1 - current_player_;
  num_moves_ += 1;
}

std::vector<Action> TicTacToeState::LegalActions() const {
  if (IsTerminal()) return {};
  // Can move in any empty cell.
  std::vector<Action> moves;
  for (int cell = 0; cell < kNumCells; ++cell) {
    if (board_[cell] == CellState::kEmpty) {
      moves.push_back(cell);
    }
  }
  return moves;
}

std::vector<Action> TicTacToeState::OriginalLegalActions(Player player)  {
  if (IsTerminal()) return {};
  // Can move in any empty cell.
  std::vector<Action> moves;
  for (int cell = 0; cell < kNumCells; ++cell) {
    if (board_[cell] == CellState::kEmpty) {
      moves.push_back(cell);
    }
  }
  return moves;
}

std::string TicTacToeState::ActionToString(Player player,
                                           Action action_id) const {
  return absl::StrCat(StateToString(PlayerToState(player)), "(",
                      action_id / kNumCols, ",", action_id % kNumCols, ")");
}

bool TicTacToeState::HasLine(Player player) const {
  CellState c = PlayerToState(player);
  return (board_[0] == c && board_[1] == c && board_[2] == c) ||
         (board_[3] == c && board_[4] == c && board_[5] == c) ||
         (board_[6] == c && board_[7] == c && board_[8] == c) ||
         (board_[0] == c && board_[3] == c && board_[6] == c) ||
         (board_[1] == c && board_[4] == c && board_[7] == c) ||
         (board_[2] == c && board_[5] == c && board_[8] == c) ||
         (board_[0] == c && board_[4] == c && board_[8] == c) ||
         (board_[2] == c && board_[4] == c && board_[6] == c);
}

bool TicTacToeState::IsFull() const { return num_moves_ == kNumCells; }

TicTacToeState::TicTacToeState(std::shared_ptr<const Game> game) : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);
}

std::string TicTacToeState::ToString() const {
  std::string str;
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      absl::StrAppend(&str, StateToString(BoardAt(r, c)));
    }
    if (r < (kNumRows - 1)) {
      absl::StrAppend(&str, "\n");
    }
  }
  return str;
}

std::string TicTacToeState::GetIDString() {
  std::string id_str;
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      absl::StrAppend(&id_str, StateToString(BoardAt(r, c)));
    }
  }
  return id_str;
}

bool TicTacToeState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || IsFull();
}

bool TicTacToeState::IsInverted() {
  return inverted_;
}

std::vector<double> TicTacToeState::Returns() const {
  if (HasLine(Player{0})) {
    return {1.0, -1.0};
  } else if (HasLine(Player{1})) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string TicTacToeState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string TicTacToeState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void TicTacToeState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{static_cast<int>(board_[cell]), cell}] = 1.0;
  }

  // print out each dimension of tensor view 
//   std::cerr << "----" << std::endl;
//   std::cerr << "----" << std::endl;
//   std::cerr << "" << std::endl;
//   std::cerr << "Obs tensor: " << std::endl;
//   for (int idx = 0; idx < 3; idx++) {
//     for (int cell = 0; cell < kNumCells; cell++) {
//       std::cerr << "idx: " << idx << ", cell: " << cell << ", view: " << view[{idx, cell}] << std::endl;
//     }
//     std::cerr << "----" << std::endl;
//   }
//   std::cerr << "" << std::endl;
//   std::cerr << "----" << std::endl;
//   std::cerr << "----" << std::endl;
}

void TicTacToeState::InvertedObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // inverted version
  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{static_cast<int>(InvertCellState(board_[cell])), cell}] = 1.0;
  }

  // print out each dimension of tensor view   
//   for (int idx = 0; idx < 3; idx++) {
//     for (int cell = 0; cell < kNumCells; cell++) {
//       std::cerr << "idx: " << idx << ", cell: " << cell << ", inverted_view: " << view[{idx, cell}] << std::endl;
//     }
//     std::cerr << "----" << std::endl;
//   }
}

void TicTacToeState::UndoAction(Player player, Action move) {
  board_[move] = CellState::kEmpty;
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_ -= 1;
  history_.pop_back();
  --move_number_;
}

void TicTacToeState::FillBoardFromStr(std::string state_str, bool inverted) {
  if (state_str.length() == kNumCells){
    int board_idx = 0;
    int num_xs = 0;
    int num_os = 0;
    if (inverted) {
        inverted_ = true;
        for(char& c : state_str) {
            if (c == '.') {
                board_[board_idx] = CellState::kEmpty;
            }
            else if (c == 'o') {
                board_[board_idx] = CellState::kCross;
                num_xs++;
            }
            else if (c == 'x') {
                board_[board_idx] = CellState::kNought;
                num_os++;
            }
            else {
                SpielFatalError("Unknown str.");
            }

            board_idx++;
        }

        // Assign correct player based on pieces on the board
        if (num_xs == num_os)
            current_player_ = 1;
        else
            current_player_ = 0;
        
        if ((num_xs > num_os) || (num_os - num_xs > 1)) {
            std::cout << "State ID: " << state_str << std::endl;
            SpielFatalError("Illegal starting state!");
        }
    } else {
        inverted_ = false;
        for(char& c : state_str) {
            if (c == '.') {
                board_[board_idx] = CellState::kEmpty;
            }
            else if (c == 'o') {
                board_[board_idx] = CellState::kNought;
                num_os++;
            }
            else if (c == 'x') {
                board_[board_idx] = CellState::kCross;
                num_xs++;
            }
            else {
                SpielFatalError("Unknown str.");
            }

            board_idx++;
        }

        // Assign correct player based on pieces on the board
        if (num_xs == num_os)
        current_player_ = 0;
        else
        current_player_ = 1;
        
        if ((num_os > num_xs) || (num_xs - num_os > 1)) {
            std::cout << "State ID: " << state_str << std::endl;
            SpielFatalError("Illegal starting state!");
        }
    }

    // Account for prior moves
    num_moves_ = num_xs + num_os;

  } else {
    SpielFatalError("State string does not match board size!");
  }
}

std::unique_ptr<State> TicTacToeState::Clone() const {
  return std::unique_ptr<State>(new TicTacToeState(*this));
}

std::vector<std::pair<std::unique_ptr<State>, std::vector<Action>>> TicTacToeState::CanonicalStates() {
// std::pair<std::vector<std::unique_ptr<State>>, std::vector<std::vector<Action>>> TicTacToeState::CanonicalStates() {
// std::vector<std::unique_ptr<State>> TicTacToeState::CanonicalStates() {
//   std::cout << "----------------------------------" << std::endl;
//   std::cout << " " << std::endl;

  std::vector<std::pair<std::unique_ptr<State>, std::vector<Action>>> states_and_actions;

  // Get current state's board and/or id string
  std::string current_state_id = GetIDString();
//   std::cout << "Original State ID: " << GetIDString() << std::endl;
//   std::cerr << "Original State: " << std::endl << ToString() << std::endl;
  std::vector<Action> original_legal_actions = LegalActions();

  // ---------------------------------------------
  // swap pieces
  // ---------------------------------------------
  std::unique_ptr<State> inv_state = Clone();
  inv_state->FillBoardFromStr(current_state_id, true);
//   std::cout << " " << std::endl;
//   std::cout << "Inv ID: " << inv_state->GetIDString() << std::endl;
//   std::cout << "Inv State: " << std::endl << inv_state->ToString() << std::endl;
  states_and_actions.emplace_back(inv_state->Clone(), inv_state->LegalActions());

  // ---------------------------------------------
  // rotate 90 degrees
  // ---------------------------------------------
  int rot_90_idxs[kNumCells] = {3,6,9,2,5,8,1,4,7};
  std::vector<int> rot_90_policy_idxs;
  std::string rot_90_str;
  for (int idx = 0; idx < kNumCells; ++idx) {
    int board_idx = rot_90_idxs[idx] - 1;
    absl::StrAppend(&rot_90_str, StateToString(board_[board_idx]));
  }

  std::unique_ptr<State> rot_90_state = Clone();
  rot_90_state->FillBoardFromStr(rot_90_str, false);
//   std::cout << " " << std::endl;
//   std::cout << "Rot 90 ID: " << rot_90_state->GetIDString() << std::endl;
//   std::cout << "Rot 90 State: " << std::endl << rot_90_state->ToString() << std::endl;

  // std::cout << "Original Legal Moves: " << absl::StrJoin(original_legal_actions, ", ") << std::endl;
  std::vector<Action> rot_90_action_idxs;
  for (auto const & act : original_legal_actions) {
    int x = std::distance(rot_90_idxs, std::find(rot_90_idxs, rot_90_idxs + kNumCells, act + 1));
    rot_90_action_idxs.push_back(x);
  }
  // std::cout << "Rot 90 Legal Moves: " << absl::StrJoin(rot_90_action_idxs, ", ") << std::endl;
  // action_permutation_indices.push_back(rot_90_action_idxs);
  states_and_actions.emplace_back(rot_90_state->Clone(), rot_90_action_idxs);

  // ---------------------------------------------
  // swap pieces, rotate 90 degrees
  // ---------------------------------------------
  std::unique_ptr<State> inv_rot_90_state = Clone();
  inv_rot_90_state->FillBoardFromStr(rot_90_str, true);
//   std::cout << " " << std::endl;
//   std::cout << "Inv Rot 90 ID: " << inv_rot_90_state->GetIDString() << std::endl;
//   std::cout << "Inv Rot 90 State: " << std::endl << inv_rot_90_state->ToString() << std::endl;
  states_and_actions.emplace_back(inv_rot_90_state->Clone(), rot_90_action_idxs);

  // ---------------------------------------------
  // rotate 180 degrees
  // ---------------------------------------------
  int rot_180_idxs[kNumCells] = {9,8,7,6,5,4,3,2,1};
  std::string rot_180_str;
  for (int idx = 0; idx < kNumCells; ++idx) {
    int board_idx = rot_180_idxs[idx] - 1;
    absl::StrAppend(&rot_180_str, StateToString(board_[board_idx]));
  }
  std::unique_ptr<State> rot_180_state = Clone();
  rot_180_state->FillBoardFromStr(rot_180_str, false);
  // canonical_states.push_back(rot_180_state->Clone());
//   std::cout << " " << std::endl;
//   std::cout << "Rot 180 ID: " << rot_180_state->GetIDString() << std::endl;
//   std::cout << "Rot 180 State: " << std::endl << rot_180_state->ToString() << std::endl;
  // std::cout << "Original Legal Moves: " << absl::StrJoin(original_legal_actions, ", ") << std::endl;
  std::vector<Action> rot_180_action_idxs;
  for (auto const & act : original_legal_actions) {
    int x = std::distance(rot_180_idxs, std::find(rot_180_idxs, rot_180_idxs + kNumCells, act + 1));
    rot_180_action_idxs.push_back(x);
  }
  // std::cout << "Rot 180 Legal Moves: " << absl::StrJoin(rot_180_action_idxs, ", ") << std::endl;
  // action_permutation_indices.push_back(rot_180_action_idxs);
  states_and_actions.emplace_back(rot_180_state->Clone(), rot_180_action_idxs);

  // ---------------------------------------------
  // swap pieces, rotate 180 degrees
  // ---------------------------------------------
  std::unique_ptr<State> inv_rot_180_state = Clone();
  inv_rot_180_state->FillBoardFromStr(rot_180_str, true);
//   std::cout << " " << std::endl;
//   std::cout << "Inv Rot 180 ID: " << inv_rot_180_state->GetIDString() << std::endl;
//   std::cout << "Inv Rot 180 State: " << std::endl << inv_rot_180_state->ToString() << std::endl;
  states_and_actions.emplace_back(inv_rot_180_state->Clone(), rot_180_action_idxs);

  // ---------------------------------------------
  // rotate 270 degrees
  // ---------------------------------------------
  int rot_270_idxs[kNumCells] = {7,4,1,8,5,2,9,6,3};
  std::string rot_270_str;
  for (int idx = 0; idx < kNumCells; ++idx) {
    int board_idx = rot_270_idxs[idx] - 1;
    absl::StrAppend(&rot_270_str, StateToString(board_[board_idx]));
  }
  std::unique_ptr<State> rot_270_state = Clone();
  rot_270_state->FillBoardFromStr(rot_270_str, false);
  // canonical_states.push_back(rot_270_state->Clone());
//   std::cout << " " << std::endl;
//   std::cout << "Rot 270 ID: " << rot_270_state->GetIDString() << std::endl;
//   std::cout << "Rot 270 State: " << std::endl << rot_270_state->ToString() << std::endl;
  // std::cout << "Original Legal Moves: " << absl::StrJoin(original_legal_actions, ", ") << std::endl;
  std::vector<Action> rot_270_action_idxs;
  for (auto const & act : original_legal_actions) {
    int x = std::distance(rot_270_idxs, std::find(rot_270_idxs, rot_270_idxs + kNumCells, act + 1));
    rot_270_action_idxs.push_back(x);
  }
  // std::cout << "Rot 270 Legal Moves: " << absl::StrJoin(rot_270_action_idxs, ", ") << std::endl;
  // action_permutation_indices.push_back(rot_270_action_idxs);
  states_and_actions.emplace_back(rot_270_state->Clone(), rot_270_action_idxs);

  // ---------------------------------------------
  // swap pieces, rotate 270 degrees
  // ---------------------------------------------
  std::unique_ptr<State> inv_rot_270_state = Clone();
  inv_rot_270_state->FillBoardFromStr(rot_270_str, true);
//   std::cout << " " << std::endl;
//   std::cout << "Inv Rot 270 ID: " << inv_rot_270_state->GetIDString() << std::endl;
//   std::cout << "Inv Rot 270 State: " << std::endl << inv_rot_270_state->ToString() << std::endl;
  states_and_actions.emplace_back(inv_rot_270_state->Clone(), rot_270_action_idxs);
  

  // ---------------------------------------------
  // vertical mirror 
  // ---------------------------------------------
  int v_mirror_idxs[kNumCells] = {7,8,9,4,5,6,1,2,3};
  std::string v_mirror_str;
  for (int idx = 0; idx < kNumCells; ++idx) {
    int board_idx = v_mirror_idxs[idx] - 1;
    absl::StrAppend(&v_mirror_str, StateToString(board_[board_idx]));
  }
  std::unique_ptr<State> v_mirror_state = Clone();
  v_mirror_state->FillBoardFromStr(v_mirror_str, false);
  // canonical_states.push_back(v_mirror_state->Clone());
//   std::cout << " " << std::endl;
//   std::cout << "V Mirror ID: " << v_mirror_state->GetIDString() << std::endl;
//   std::cout << "V Mirror State: " << std::endl << v_mirror_state->ToString() << std::endl;
  // std::cout << "Original Legal Moves: " << absl::StrJoin(original_legal_actions, ", ") << std::endl;
  std::vector<Action> v_mirror_action_idxs;
  for (auto const & act : original_legal_actions) {
    int x = std::distance(v_mirror_idxs, std::find(v_mirror_idxs, v_mirror_idxs + kNumCells, act + 1));
    v_mirror_action_idxs.push_back(x);
  }
  // std::cout << "V Mirror Legal Moves: " << absl::StrJoin(v_mirror_action_idxs, ", ") << std::endl;
  // action_permutation_indices.push_back(v_mirror_action_idxs);
  states_and_actions.emplace_back(v_mirror_state->Clone(), v_mirror_action_idxs);

  // ---------------------------------------------
  // swap pieces, vertical mirror 
  // ---------------------------------------------
  std::unique_ptr<State> inv_v_mirror_state = Clone();
  inv_v_mirror_state->FillBoardFromStr(v_mirror_str, true);
//   std::cout << " " << std::endl;
//   std::cout << "Inv V Mirror ID: " << inv_v_mirror_state->GetIDString() << std::endl;
//   std::cout << "Inv V Mirror State: " << std::endl << inv_v_mirror_state->ToString() << std::endl;
  states_and_actions.emplace_back(inv_v_mirror_state->Clone(), v_mirror_action_idxs);

  // ---------------------------------------------
  // horizontal mirror 
  // ---------------------------------------------
  int h_mirror_idxs[kNumCells] = {3,2,1,6,5,4,9,8,7};
  std::string h_mirror_str;
  for (int idx = 0; idx < kNumCells; ++idx) {
    int board_idx = h_mirror_idxs[idx] - 1;
    absl::StrAppend(&h_mirror_str, StateToString(board_[board_idx]));
  }
  std::unique_ptr<State> h_mirror_state = Clone();
  h_mirror_state->FillBoardFromStr(h_mirror_str, false);
  // canonical_states.push_back(h_mirror_state->Clone());
//   std::cout << " " << std::endl;
//   std::cout << "H Mirror ID: " << h_mirror_state->GetIDString() << std::endl;
//   std::cout << "H Mirror State: " << std::endl << h_mirror_state->ToString() << std::endl;
  // std::cout << "Original Legal Moves: " << absl::StrJoin(original_legal_actions, ", ") << std::endl;
  std::vector<Action> h_mirror_action_idxs;
  for (auto const & act : original_legal_actions) {
    int x = std::distance(h_mirror_idxs, std::find(h_mirror_idxs, h_mirror_idxs + kNumCells, act + 1));
    h_mirror_action_idxs.push_back(x);
  }
  // std::cout << "H Mirror Legal Moves: " << absl::StrJoin(h_mirror_action_idxs, ", ") << std::endl;
  // action_permutation_indices.push_back(h_mirror_action_idxs);
  states_and_actions.emplace_back(h_mirror_state->Clone(), h_mirror_action_idxs);

  // ---------------------------------------------
  // swap pieces, horizontal mirror 
  // ---------------------------------------------
  std::unique_ptr<State> inv_h_mirror_state = Clone();
  inv_h_mirror_state->FillBoardFromStr(h_mirror_str, true);
//   std::cout << " " << std::endl;
//   std::cout << "Inv H Mirror ID: " << inv_h_mirror_state->GetIDString() << std::endl;
//   std::cout << "Inv H Mirror State: " << std::endl << inv_h_mirror_state->ToString() << std::endl;
  states_and_actions.emplace_back(inv_h_mirror_state->Clone(), h_mirror_action_idxs);

  // ---------------------------------------------
  // rotate 90 degrees, vertical mirror 
  // ---------------------------------------------
  int rot_90_v_mirror_idxs[kNumCells] = {1,4,7,2,5,8,3,6,9};
  std::string rot_90_v_mirror_str;
  for (int idx = 0; idx < kNumCells; ++idx) {
    int board_idx = rot_90_v_mirror_idxs[idx] - 1;
    absl::StrAppend(&rot_90_v_mirror_str, StateToString(board_[board_idx]));
  }
  std::unique_ptr<State> rot_90_v_mirror_state = Clone();
  rot_90_v_mirror_state->FillBoardFromStr(rot_90_v_mirror_str, false);
  // canonical_states.push_back(rot_90_v_mirror_state->Clone());
//   std::cout << " " << std::endl;
//   std::cout << "Rot 90 V Mirror ID: " << rot_90_v_mirror_state->GetIDString() << std::endl;
//   std::cout << "Rot 90 V Mirror State: " << std::endl << rot_90_v_mirror_state->ToString() << std::endl;
  // std::cout << "Original Legal Moves: " << absl::StrJoin(original_legal_actions, ", ") << std::endl;
  std::vector<Action> rot_90_v_mirror_action_idxs;
  for (auto const & act : original_legal_actions) {
    int x = std::distance(rot_90_v_mirror_idxs, std::find(rot_90_v_mirror_idxs, rot_90_v_mirror_idxs + kNumCells, act + 1));
    rot_90_v_mirror_action_idxs.push_back(x);
  }
  // std::cout << "Rot 90 V Mirror Legal Moves: " << absl::StrJoin(rot_90_v_mirror_action_idxs, ", ") << std::endl;
  // action_permutation_indices.push_back(rot_90_v_mirror_action_idxs);
  states_and_actions.emplace_back(rot_90_v_mirror_state->Clone(), rot_90_v_mirror_action_idxs);

  // ---------------------------------------------
  // swap pieces, rotate 90 degrees, vertical mirror 
  // ---------------------------------------------
  std::unique_ptr<State> inv_rot_90_v_mirror_state = Clone();
  inv_rot_90_v_mirror_state->FillBoardFromStr(rot_90_v_mirror_str, true);
//   std::cout << " " << std::endl;
//   std::cout << "Inv Rot 90 V Mirror ID: " << inv_rot_90_v_mirror_state->GetIDString() << std::endl;
//   std::cout << "Inv Rot 90 V Mirror State: " << std::endl << inv_rot_90_v_mirror_state->ToString() << std::endl;
  states_and_actions.emplace_back(inv_rot_90_v_mirror_state->Clone(), rot_90_v_mirror_action_idxs);

  // ---------------------------------------------
  // rotate 90 degrees, horizontal mirror 
  // ---------------------------------------------
  int rot_90_h_mirror_idxs[kNumCells] = {9,6,3,8,5,2,7,4,1};
  std::string rot_90_h_mirror_str;
  for (int idx = 0; idx < kNumCells; ++idx) {
    int board_idx = rot_90_h_mirror_idxs[idx] - 1;
    absl::StrAppend(&rot_90_h_mirror_str, StateToString(board_[board_idx]));
  }
  std::unique_ptr<State> rot_90_h_mirror_state = Clone();
  rot_90_h_mirror_state->FillBoardFromStr(rot_90_h_mirror_str, false);
  // canonical_states.push_back(rot_90_h_mirror_state->Clone());
//   std::cout << " " << std::endl;
//   std::cout << "Rot 90 H Mirror ID: " << rot_90_h_mirror_state->GetIDString() << std::endl;
//   std::cout << "Rot 90 H Mirror State: " << std::endl << rot_90_h_mirror_state->ToString() << std::endl;
  // std::cout << "Original Legal Moves: " << absl::StrJoin(original_legal_actions, ", ") << std::endl;
  std::vector<Action> rot_90_h_mirror_action_idxs;
  for (auto const & act : original_legal_actions) {
    int x = std::distance(rot_90_h_mirror_idxs, std::find(rot_90_h_mirror_idxs, rot_90_h_mirror_idxs + kNumCells, act + 1));
    rot_90_h_mirror_action_idxs.push_back(x);
  }
  // std::cout << "Rot 90 H Mirror Legal Moves: " << absl::StrJoin(rot_90_h_mirror_action_idxs, ", ") << std::endl;
  // action_permutation_indices.push_back(rot_90_h_mirror_action_idxs);
  states_and_actions.emplace_back(rot_90_h_mirror_state->Clone(), rot_90_h_mirror_action_idxs);

  // ---------------------------------------------
  // swap pieces, rotate 90 degrees, horizontal mirror 
  // ---------------------------------------------
  std::unique_ptr<State> inv_rot_90_h_mirror_state = Clone();
  inv_rot_90_h_mirror_state->FillBoardFromStr(rot_90_h_mirror_str, true);
//   std::cout << " " << std::endl;
//   std::cout << "Inv Rot 90 H Mirror ID: " << inv_rot_90_h_mirror_state->GetIDString() << std::endl;
//   std::cout << "Inv Rot 90 H Mirror State: " << std::endl << inv_rot_90_h_mirror_state->ToString() << std::endl;
  states_and_actions.emplace_back(inv_rot_90_h_mirror_state->Clone(), rot_90_h_mirror_action_idxs);
  
//   SpielFatalError("Test!");

//   return canonical_states;
//   return std::make_pair(std::move(canonical_states), action_permutation_indices);
  return states_and_actions;
}

TicTacToeGame::TicTacToeGame(const GameParameters& params)
    : Game(kGameType, params) {}

std::unique_ptr<State> TicTacToeGame::NewInitialState(const std::string& str) const {

//   std::string state_str = "xoo..x...";
  TicTacToeState init_state = TicTacToeState(shared_from_this());
  init_state.FillBoardFromStr(str, false);

//   std::cout << "State ID: " << state_str << std::endl;
//   std::cout << "Board: " << init_state.ToString() << std::endl;
  return std::unique_ptr<State>(init_state.Clone());
}

}  // namespace tic_tac_toe
}  // namespace open_spiel
