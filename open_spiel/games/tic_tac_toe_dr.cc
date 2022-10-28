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

#include "open_spiel/games/tic_tac_toe_dr.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <time.h>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace tic_tac_toe_dr {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"tic_tac_toe_dr",
    /*long_name=*/"Tic Tac Toe DR",
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
  return std::shared_ptr<const Game>(new TicTacToeDRGame(params));
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

// CellState StringToState(const char *cell_char) {
//     std::cout << "input char: " << cell_char << std::endl;
//     bool temp = cell_char == ".";
//     std::cout << "char match: " << temp << std::endl;
    
//     if (cell_char == ".")
//       return CellState::kEmpty;
//     else if (cell_char == "o")
//       return CellState::kNought;
//     else if (cell_char == "x")
//       return CellState::kCross;
//     else
//       SpielFatalError("Unknown str.");
// }

void TicTacToeDRState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(board_[move], CellState::kEmpty);
  board_[move] = PlayerToState(CurrentPlayer());
  if (HasLine(current_player_)) {
    outcome_ = current_player_;
  }
  current_player_ = 1 - current_player_;
  num_moves_ += 1;
}

std::vector<Action> TicTacToeDRState::LegalActions() const {
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

std::vector<Action> TicTacToeDRState::OriginalLegalActions(Player player)  {
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

std::string TicTacToeDRState::ActionToString(Player player,
                                           Action action_id) const {
  return absl::StrCat(StateToString(PlayerToState(player)), "(",
                      action_id / kNumCols, ",", action_id % kNumCols, ")");
}

bool TicTacToeDRState::HasLine(Player player) const {
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

bool TicTacToeDRState::IsFull() const {return num_moves_ == kNumCells; }

TicTacToeDRState::TicTacToeDRState(std::shared_ptr<const Game> game) : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);
}

std::string TicTacToeDRState::ToString() const {
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

std::string TicTacToeDRState::GetIDString() {
  std::string id_str;
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      absl::StrAppend(&id_str, StateToString(BoardAt(r, c)));
    }
  }
  return id_str;
}

bool TicTacToeDRState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || IsFull();
}

std::vector<double> TicTacToeDRState::Returns() const {
  if (HasLine(Player{0})) {
    return {1.0, -1.0};
  } else if (HasLine(Player{1})) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string TicTacToeDRState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string TicTacToeDRState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void TicTacToeDRState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    // std::cerr << "Cell: " << cell << ", Value: " << board_[cell] << ", Value (int): " << static_cast<int>(board_[cell]) << std::endl;
    view[{static_cast<int>(board_[cell]), cell}] = 1.0;
  }

  // print out each dimension of tensor view   
//   for (int idx = 0; idx < 3; idx++) {
    // for (int cell = 0; cell < kNumCells; cell++) {
    //   std::cerr << "idx: " << idx << ", cell: " << cell << ", view: " << view[{static_cast<int>(board_[idx]), cell}] << std::endl;
    //   std::cerr << "idx: " << idx << ", cell: " << cell << ", view: " << view[{idx, cell}] << std::endl;
    
    // }
    // std::cerr << "----" << std::endl;
//   }
}

void TicTacToeDRState::UndoAction(Player player, Action move) {
  board_[move] = CellState::kEmpty;
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_ -= 1;
  history_.pop_back();
  --move_number_;
}

void TicTacToeDRState::FillBoardFromStr(std::string state_str) {
  if (state_str.length() == kNumCells){
    int board_idx = 0;
    int num_xs = 0;
    int num_os = 0;
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

    // Account for prior moves
    num_moves_ = num_xs + num_os;

  } else {
    SpielFatalError("State string does not match board size!");
  }
}

std::unique_ptr<State> TicTacToeDRState::Clone() const {
  return std::unique_ptr<State>(new TicTacToeDRState(*this));
}

TicTacToeDRGame::TicTacToeDRGame(const GameParameters& params)
    : Game(kGameType, params) {
  srand(time(NULL));
  std::ifstream file_in("data/tic_tac_toe_states.txt");

  // Check if object is valid
  if(!file_in)
  {
    SpielFatalError("Cannot open state strings file!");
  }

  std::string line;
  while (std::getline(file_in, line)) {
    if(line.size() > 0) {
      state_strs.push_back(line);
    }
    else {
      SpielFatalError("Empty state string!");
    }
  }
  file_in.close();
}

std::unique_ptr<State> TicTacToeDRGame::NewInitialState() const {
  int state_idx = rand() % state_strs.size();
  std::string state_str = state_strs[state_idx];

//   for (int idx = 0; idx < state_strs.size(); ++idx) {
    // std::cout << "State ID: " << state_strs[idx] << std::endl;
//   }
  
  TicTacToeDRState init_state = TicTacToeDRState(shared_from_this());
//   std::string state_str = "xoo..x...";
  init_state.FillBoardFromStr(state_str);

//   std::cout << "State ID: " << state_str << std::endl;
//   std::cout << "Board: " << init_state.ToString() << std::endl;



//   return std::unique_ptr<State>(new TicTacToeDRState(shared_from_this()));
  return std::unique_ptr<State>(init_state.Clone());
}

}  // namespace tic_tac_toe_dr
}  // namespace open_spiel
