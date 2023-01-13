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

#include "open_spiel/games/connect_four_turns.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace connect_four_turns {
namespace {

// Facts about the game
const GameType kGameType{
    /*short_name=*/"connect_four_turns",
    /*long_name=*/"Connect Four Turns",
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
  return std::shared_ptr<const Game>(new ConnectFourTurnsGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kCross;
    case 1:
      return CellState::kNought;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
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
      return "This will never return.";
  }
}
}  // namespace

CellState& ConnectFourTurnsState::CellAt(int row, int col) {
  return board_[row * kCols + col];
}

CellState ConnectFourTurnsState::CellAt(int row, int col) const {
  return board_[row * kCols + col];
}

int ConnectFourTurnsState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return current_player_;
  }
}

void ConnectFourTurnsState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(CellAt(kRows - 1, move), CellState::kEmpty);
  int row = 0;
  while (CellAt(row, move) != CellState::kEmpty) ++row;
  CellAt(row, move) = PlayerToState(CurrentPlayer());

  if (HasLine(current_player_)) {
    outcome_ = static_cast<Outcome>(current_player_);
  } else if (IsFull()) {
    outcome_ = Outcome::kDraw;
  }

  current_player_ = 1 - current_player_;
}

std::vector<Action> ConnectFourTurnsState::LegalActions() const {
  // Can move in any non-full column.
  std::vector<Action> moves;
  if (IsTerminal()) return moves;
  for (int col = 0; col < kCols; ++col) {
    if (CellAt(kRows - 1, col) == CellState::kEmpty) moves.push_back(col);
  }
  return moves;
}

std::string ConnectFourTurnsState::ActionToString(Player player,
                                             Action action_id) const {
  return absl::StrCat(StateToString(PlayerToState(player)), action_id);
}

bool ConnectFourTurnsState::HasLineFrom(Player player, int row, int col) const {
  return HasLineFromInDirection(player, row, col, 0, 1) ||
         HasLineFromInDirection(player, row, col, -1, -1) ||
         HasLineFromInDirection(player, row, col, -1, 0) ||
         HasLineFromInDirection(player, row, col, -1, 1);
}

bool ConnectFourTurnsState::HasLineFromInDirection(Player player, int row, int col,
                                              int drow, int dcol) const {
  if (row + 3 * drow >= kRows || col + 3 * dcol >= kCols ||
      row + 3 * drow < 0 || col + 3 * dcol < 0)
    return false;
  CellState c = PlayerToState(player);
  for (int i = 0; i < 4; ++i) {
    if (CellAt(row, col) != c) return false;
    row += drow;
    col += dcol;
  }
  return true;
}

bool ConnectFourTurnsState::HasLine(Player player) const {
  CellState c = PlayerToState(player);
  for (int col = 0; col < kCols; ++col) {
    for (int row = 0; row < kRows; ++row) {
      if (CellAt(row, col) == c && HasLineFrom(player, row, col)) return true;
    }
  }
  return false;
}

bool ConnectFourTurnsState::IsFull() const {
  for (int col = 0; col < kCols; ++col) {
    if (CellAt(kRows - 1, col) == CellState::kEmpty) return false;
  }
  return true;
}

ConnectFourTurnsState::ConnectFourTurnsState(std::shared_ptr<const Game> game)
    : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);
}

std::string ConnectFourTurnsState::ToString() const {
  std::string str;
  for (int row = kRows - 1; row >= 0; --row) {
    for (int col = 0; col < kCols; ++col) {
      str.append(StateToString(CellAt(row, col)));
    }
    str.append("\n");
  }
  return str;
}

std::string ConnectFourTurnsState::GetIDString() {
  std::string id_str;
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; ++c) {
      absl::StrAppend(&id_str, StateToString(CellAt(r, c)));
    }
  }
  return id_str;
}

bool ConnectFourTurnsState::IsTerminal() const {
  return outcome_ != Outcome::kUnknown;
}

std::vector<double> ConnectFourTurnsState::Returns() const {
  if (outcome_ == Outcome::kPlayer1) return {1.0, -1.0};
  if (outcome_ == Outcome::kPlayer2) return {-1.0, 1.0};
  return {0.0, 0.0};
}

std::string ConnectFourTurnsState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string ConnectFourTurnsState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

int PlayerRelative(CellState state, Player current) {
  switch (state) {
    case CellState::kNought:
      return current == 0 ? 0 : 1;
    case CellState::kCross:
      return current == 1 ? 0 : 1;
    case CellState::kEmpty:
      return 2;
    default:
      SpielFatalError("Unknown player type.");
  }
}

void ConnectFourTurnsState::ObservationTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<2> view(values, {kCellStates+1, kNumCells}, true);

  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{PlayerRelative(board_[cell], player), cell}] = 1.0;
  }

  // current player plane
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{3, cell}] = current_player_;
  }

}

void ConnectFourTurnsState::FillBoardFromStr(std::string state_str, bool inverted) {
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

std::unique_ptr<State> ConnectFourTurnsState::Clone() const {
  return std::unique_ptr<State>(new ConnectFourTurnsState(*this));
}

std::vector<std::pair<std::unique_ptr<State>, std::vector<Action>>> ConnectFourTurnsState::CanonicalStates() {
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
  int rot_90_idxs[kNumCells] = {7,14,21,28,35,42,6,13,20,27,34,41,5,12,19,26,33,40,4,11,18,25,32,39,3,10,17,24,31,38,2,9,16,23,30,37,1,8,15,22,29,36};
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
  int rot_180_idxs[kNumCells] = {42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};
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
  int rot_270_idxs[kNumCells] = {36,29,22,15,8,1,37,30,23,16,9,2,38,31,24,17,10,3,39,32,25,18,11,4,40,33,26,19,12,5,41,34,27,20,13,6,42,35,28,21,14,7};
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
  int v_mirror_idxs[kNumCells] = {36,37,38,39,40,41,42,29,30,31,32,33,34,35,22,23,24,25,26,27,28,15,16,17,18,19,20,21,8,9,10,11,12,13,14,1,2,3,4,5,6,7};
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
  int h_mirror_idxs[kNumCells] = {7,6,5,4,3,2,1,14,13,12,11,10,9,8,21,20,19,18,17,16,15,28,27,26,25,24,23,22,35,34,33,32,31,30,29,42,41,40,39,38,37,36};
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
  int rot_90_v_mirror_idxs[kNumCells] = {1,8,15,22,29,36,2,9,16,23,30,37,3,10,17,24,31,38,4,11,18,25,32,39,5,12,19,26,33,40,6,13,20,27,34,41,7,14,21,28,35,42};
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
  int rot_90_h_mirror_idxs[kNumCells] = {42,35,28,21,14,7,41,34,27,20,13,6,40,33,26,19,12,5,39,32,25,18,11,4,38,31,24,17,10,3,37,30,23,16,9,2,36,29,22,15,8,1};
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

ConnectFourTurnsGame::ConnectFourTurnsGame(const GameParameters& params)
    : Game(kGameType, params) {}

ConnectFourTurnsState::ConnectFourTurnsState(std::shared_ptr<const Game> game,
                                   const std::string& str)
    : State(game) {
  int xs = 0;
  int os = 0;
  int r = 5;
  int c = 0;
  for (const char ch : str) {
    switch (ch) {
      case '.':
        CellAt(r, c) = CellState::kEmpty;
        break;
      case 'x':
        ++xs;
        CellAt(r, c) = CellState::kCross;
        break;
      case 'o':
        ++os;
        CellAt(r, c) = CellState::kNought;
        break;
    }
    if (ch == '.' || ch == 'x' || ch == 'o') {
      ++c;
      if (c >= kCols) {
        r--;
        c = 0;
      }
    }
  }
  SPIEL_CHECK_TRUE(xs == os || xs == (os + 1));
  SPIEL_CHECK_TRUE(r == -1 && ("Problem parsing state (incorrect rows)."));
  SPIEL_CHECK_TRUE(c == 0 &&
                   ("Problem parsing state (column value should be 0)"));
  current_player_ = (xs == os) ? 0 : 1;

  if (HasLine(0)) {
    outcome_ = Outcome::kPlayer1;
  } else if (HasLine(1)) {
    outcome_ = Outcome::kPlayer2;
  } else if (IsFull()) {
    outcome_ = Outcome::kDraw;
  }
}

}  // namespace connect_four_turns
}  // namespace open_spiel
