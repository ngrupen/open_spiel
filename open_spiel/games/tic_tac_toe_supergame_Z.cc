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

#include "open_spiel/games/tic_tac_toe_supergame_Z.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace tic_tac_toe_supergame_Z {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"tic_tac_toe_supergame_Z",
    /*long_name=*/"Tic Tac Toe (SuperGame Z)",
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
  return std::shared_ptr<const Game>(new TicTacToeSuperGameZGame(params));
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



std::string ActionToBase(Action action_id, int base) {
  std::string base_str;

//   std::cerr << "Action: " << action_id << std::endl;
//   std::cerr << "Base: " << base << std::endl;

  std::string digs = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

//   std::cerr << "Digs: " << digs << std::endl;

  int sign;
  if (action_id < 0) {
    sign = -1;
  } else if (action_id == 0) {
    base_str.push_back(digs[0]);
    return base_str;
  } else {
    sign = 1;
  }

//   std::cerr << "Sign: " << sign << std::endl;

  Action x = action_id * sign;

  while (x) {
      base_str.push_back(digs[int(x % base)]);
      x = x/base;
  }

//   std::cerr << "Base str backwards: " << base_str << std::endl;
  
  std::reverse(base_str.begin(), base_str.end());

//   std::cerr << "Base str: " << base_str << std::endl;

  return base_str;

}

std::string ActionToBaseFixed(Action action_id, int base, int width) {

  std::string base_str = ActionToBase(action_id, 3);
//   std::cerr << "base str: " << base_str << std::endl;

//   auto new_str = std::string(n_zero - std::min(n_zero, old_str.length()), '0') + base_str;
//   std::string dest = std::string( number_of_zeros, '0').append(base_str);

  base_str.insert(base_str.begin(), width - base_str.length(), '0');

  return base_str;

}

std::array<CellState, kNumCells> GetBoardFromAction(Action action_id) {
  std::string board_str = ActionToBaseFixed(action_id, 3, kNumCells);
//   std::cerr << "board str fixed: " << board_str << std::endl;


  std::array<CellState, kNumCells> new_board;
  std::fill(begin(new_board), end(new_board), CellState::kEmpty);

  CellState markers[3] = { CellState::kNought, CellState::kEmpty, CellState::kCross };

  for (int cell = 0; cell < board_str.length(); ++cell) {
    // std::cerr << "i: " << cell << std::endl;
    std::string idx(1, board_str[cell]);
    // std::cerr << "board string i: " << std::stoi(idx) << std::endl;
    // std::cerr << "marker i: " << markers[std::stoi(idx)] << std::endl;

    new_board[cell] = markers[std::stoi(idx)] ;
    // std::cerr << "board cell i: " << new_board[cell] << std::endl;
  }

  return new_board;
}

std::string BoardStateToString(std::array<CellState, kNumCells> board) {
  std::vector<int> board_str;
  for (int cell = 0; cell < kNumCells; ++cell) {
    if (board[cell] == CellState::kCross) {
      board_str.push_back(1);
    } else if (board[cell] == CellState::kEmpty) {
      board_str.push_back(0);
    } else {
      board_str.push_back(-1);
    }
  }
  return absl::StrJoin(board_str, ",");
}


bool IsValidMove(std::string current_board, std::string next_board) {
  std::vector<std::string> current_board_parts = absl::StrSplit(current_board, ',');
  std::vector<std::string> next_board_parts = absl::StrSplit(next_board, ',');

  int changed_count = 0;
  int new_count = 0;
  int num_xs = 0;
  int num_os = 0;
  for (int i = 0; i < current_board_parts.size(); ++i) {
    if (current_board_parts[i] == "1") {
      num_xs++;
    } else if (current_board_parts[i] == "-1") {
      num_os++;
    }
  }

//   std::cerr << "num xs: " << num_xs << std::endl;
//   std::cerr << "num os: " << num_os << std::endl;

  std::string correct_move;
  if (num_xs == num_os) {
    correct_move = "1";
  } else {
    correct_move = "-1";
  }
//   std::cerr << "correct move: " << correct_move << std::endl;

  std::string shape;
  for (int i = 0; i < current_board_parts.size(); ++i) {
    if ((current_board_parts[i] == "0") && (next_board_parts[i] != "0")) {
      new_count++;
      shape = next_board_parts[i];
    } 
    else if ((current_board_parts[i] != "0") && (next_board_parts[i] != current_board_parts[i])) {
        changed_count++;
    }
  }

//   std::cerr << "shape: " << shape << std::endl;
//   std::cerr << "changed count: " << changed_count << std::endl;
//   std::cerr << "new count: " << new_count << std::endl;

  return ((changed_count == 0) && (new_count == 1) && (shape == correct_move));

}

void TicTacToeSuperGameZState::DoApplyAction(Action move) {
  std::array<CellState, kNumCells>  new_board = GetBoardFromAction(move);
  
  std::string current_board_str = BoardStateToString(board_);
//   std::cerr << "current board str: " << current_board_str << std::endl;
  std::string new_board_str = BoardStateToString(new_board);
//   std::cerr << "new board str: " << new_board_str << std::endl;

  bool valid_move = IsValidMove(current_board_str, new_board_str);
//   std::cerr << "valid: " << valid_move << std::endl;

  if (valid_move) {
    board_ = new_board;
    if (HasLine(current_player_)) {
      outcome_ = current_player_;
    }
  } else {
      invalid_mover_ = current_player_;
      outcome_ = 1 - current_player_;
  }
  current_player_ = 1 - current_player_;
  num_moves_ += 1;
}

std::vector<Action> TicTacToeSuperGameZState::LegalActions() const {
  if (IsTerminal()) return {};
  // can create any tic-tac-toe board
  std::vector<Action> moves;
  for (int cell = 0; cell < pow(3, kNumCells); ++cell) {
    moves.push_back(cell);
  }
//   std::cerr << "Legal moves: " << absl::StrJoin(moves, ",") << std::endl;
  return moves;
}

std::string TicTacToeSuperGameZState::ActionToString(Player player,
                                           Action action_id) const {
  return absl::StrCat(StateToString(PlayerToState(player)), "(",
                      action_id / kNumCols, ",", action_id % kNumCols, ")");
}

bool TicTacToeSuperGameZState::HasLine(Player player) const {
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

bool TicTacToeSuperGameZState::IsFull() const { return num_moves_ == kNumCells; }

TicTacToeSuperGameZState::TicTacToeSuperGameZState(std::shared_ptr<const Game> game) : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);
}

std::string TicTacToeSuperGameZState::ToString() const {
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

bool TicTacToeSuperGameZState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || IsFull();
}

std::vector<double> TicTacToeSuperGameZState::Returns() const {
  if ((HasLine(Player{0})) || (invalid_mover_ == Player{1})) {
    return {1.0, -1.0};
  } else if ((HasLine(Player{1})) || (invalid_mover_ == Player{0})) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

// std::vector<double> TicTacToeNoRulesState::Returns() const {
//   if (HasLine(Player{0})) {
//     return {1.0, -1.0};
//   } else if (HasLine(Player{1})) {
//     return {-1.0, 1.0};
//   } else {
//     return {0.0, 0.0};
//   }
// }

std::string TicTacToeSuperGameZState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string TicTacToeSuperGameZState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void TicTacToeSuperGameZState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{static_cast<int>(board_[cell]), cell}] = 1.0;
  }
}

void TicTacToeSuperGameZState::UndoAction(Player player, Action move) {
  board_[move] = CellState::kEmpty;
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_ -= 1;
  history_.pop_back();
  --move_number_;
}

std::unique_ptr<State> TicTacToeSuperGameZState::Clone() const {
  return std::unique_ptr<State>(new TicTacToeSuperGameZState(*this));
}

TicTacToeSuperGameZGame::TicTacToeSuperGameZGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace tic_tac_toe_supergame_Z
}  // namespace open_spiel
