// SPDX-License-Identifier: Apache-2.0
// Adapter only. The detector implementation is compiled from upstream unchanged.
#include "runtime/components/stop_token_detector.h"
#include <array>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  using litert::lm::StopTokenDetector;
  if (argc > 1 && std::string(argv[1]) == "bench") {
    std::string mode = argv[2];
    const bool nested = mode.rfind("nested-", 0) == 0;
    if (nested) mode = mode.substr(7);
    std::vector<std::vector<int>> rows(4, std::vector<int>(1));
    const size_t rounds = std::stoull(argv[3]);
    StopTokenDetector d(4);
    for (const auto& s : std::vector<std::vector<int>>{{7,8,9}, {4,4,5}, {99}}) {
      if (!d.AddStopTokenSequence(s).ok()) return 1;
    }
    size_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (size_t round = 0; round < rounds; ++round) {
      d.ResetBatch();
      for (int step = 0; step < 64; ++step) {
        std::array<int, 4> tokens;
        for (int i = 0; i < 4; ++i) {
          if (mode == "no-match") tokens[i] = (step + i) % 4;
          else if (mode == "partial") tokens[i] = std::array<int,3>{7,8,0}[(step+i)%3];
          else if (mode == "early-stop") tokens[i] = (step + i) % 13;
          else return 2;
        }
        if (nested) {
          for (size_t i = 0; i < rows.size(); ++i) rows[i][0] = tokens[i];
          if (!d.ProcessTokens(rows).ok()) return 1;
        } else if (!d.ProcessTokens(absl::MakeSpan(tokens)).ok()) return 1;
        for (int count : d.GetStepsBeforeStopTokens()) checksum += count;
        checksum += d.MaxPartialStopTokenLength(0);
      }
    }
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << ns << ' ' << checksum << '\n';
    return 0;
  }
  auto d = std::make_unique<StopTokenDetector>(1);
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream input(line);
    char command;
    input >> command;
    std::vector<int> values;
    int value;
    while (input >> value) values.push_back(value);
    absl::Status status;
    if (command == 'N') d = std::make_unique<StopTokenDetector>(values.at(0));
    else if (command == 'R') d->ResetBatch(values.at(0));
    else if (command == 'A') status = d->AddStopTokenSequence(values);
    else if (command == 'P') status = d->ProcessTokens(absl::MakeSpan(values));
#ifdef RT_RS_CURRENT_MAIN
    else if (command == 'M') {
      if (!d->MarkForRollback(values.at(0))) status = absl::NotFoundError("missing singleton");
    }
#endif
    else if (command == 'V') {
      std::vector<std::vector<int>> rows;
      for (size_t offset = 0; offset < values.size();) {
        const size_t length = values.at(offset++);
        std::vector<int> row;
        for (size_t j = 0; j < length; ++j) row.push_back(values.at(offset++));
        rows.push_back(std::move(row));
      }
      status = d->ProcessTokens(rows);
    }
    else return 2;
    std::cout << static_cast<int>(status.code()) << ' ' << d->AllDone().value();
    for (size_t i = 0; i < d->GetStopTokensFound().size(); ++i) {
      std::cout << ' ' << d->GetStopTokensFound()[i] << ' '
                << d->GetStepsBeforeStopTokens()[i] << ' '
                << d->MaxPartialStopTokenLength(i);
#ifdef RT_RS_CURRENT_MAIN
      std::cout << ' ' << d->EndsWithRollbackToken(i);
#endif
    }
    std::cout << '\n';
  }
}
