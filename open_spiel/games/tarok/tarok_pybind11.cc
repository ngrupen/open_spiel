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

#include "open_spiel/games/tarok.h"
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"

namespace open_spiel {

namespace py = ::pybind11;

void init_pyspiel_tarok(py::module& m) {
  // state object
  py::class_<tarok::TarokState, State> tarok_state(m, "TarokState");
  tarok_state.def("card_action_to_string",
                  &tarok::TarokState::CardActionToString);
  tarok_state.def("current_game_phase", &tarok::TarokState::CurrentGamePhase);
  tarok_state.def("player_cards", &tarok::TarokState::PlayerCards);
  tarok_state.def("selected_contract",
                  &tarok::TarokState::SelectedContractName);
  tarok_state.def("talon", &tarok::TarokState::Talon);
  tarok_state.def("talon_sets", &tarok::TarokState::TalonSets);
  tarok_state.def("trick_cards", &tarok::TarokState::TrickCards);
  tarok_state.def("captured_mond_penalties",
                  &tarok::TarokState::CapturedMondPenalties);
  tarok_state.def("scores_without_captured_mond_penalties",
                  &tarok::TarokState::ScoresWithoutCapturedMondPenalties);
  tarok_state.def(py::pickle(
      [](const tarok::TarokState& state) {  // __getstate__
        return SerializeGameAndState(*state.GetGame(), state);
      },
      [](const std::string& data) {  // __setstate__
        std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
            game_and_state = DeserializeGameAndState(data);
        return std::move(std::unique_ptr<tarok::TarokState>{
            static_cast<tarok::TarokState*>(game_and_state.second.release())});
      }));

  // game phase object
  py::enum_<tarok::GamePhase> game_phase(m, "TarokGamePhase");
  game_phase.value("CARD_DEALING", tarok::GamePhase::kCardDealing);
  game_phase.value("BIDDING", tarok::GamePhase::kBidding);
  game_phase.value("KING_CALLING", tarok::GamePhase::kKingCalling);
  game_phase.value("TALON_EXCHANGE", tarok::GamePhase::kTalonExchange);
  game_phase.value("TRICKS_PLAYING", tarok::GamePhase::kTricksPlaying);
  game_phase.value("FINISHED", tarok::GamePhase::kFinished);

  // contract name object
  py::enum_<tarok::ContractName> contract(m, "TarokContract");
  contract.value("KLOP", tarok::ContractName::kKlop);
  contract.value("THREE", tarok::ContractName::kThree);
  contract.value("TWO", tarok::ContractName::kTwo);
  contract.value("ONE", tarok::ContractName::kOne);
  contract.value("SOLO_THREE", tarok::ContractName::kSoloThree);
  contract.value("SOLO_TWO", tarok::ContractName::kSoloTwo);
  contract.value("SOLO_ONE", tarok::ContractName::kSoloOne);
  contract.value("BEGGAR", tarok::ContractName::kBeggar);
  contract.value("SOLO_WITHOUT", tarok::ContractName::kSoloWithout);
  contract.value("OPEN_BEGGAR", tarok::ContractName::kOpenBeggar);
  contract.value("COLOUR_VALAT_WITHOUT",
                 tarok::ContractName::kColourValatWithout);
  contract.value("VALAT_WITHOUT", tarok::ContractName::kValatWithout);
  contract.value("NOT_SELECTED", tarok::ContractName::kNotSelected);
}

}  // namespace open_spiel
