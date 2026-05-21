# AI Agent Instruction Guidelines (AGENTS.md)

This file defines the strict architectural guardrails, coding standards, and styling guidelines for all AI agents contributing to the `golfcar-tracker` repository. Agents must read and adhere to these constraints for every code generation task.

---

## 1. Project Overview & Context

`golfcar-tracker` is a hybrid C++ and Python project. 
*   **`cpp/`**: Handles performance-critical tasks (e.g., real-time tracking algorithms, math, sensor processing).
*   **`python/`**: Handles higher-level application logic, scripting, rapid prototyping, and orchestration.

---

## 2. C++ Coding Guidelines (`cpp/`)

When generating or modifying C++ code, prioritize modern, efficient, and expressive design patterns introduced in **C++20**.

### C++20 Features First
*   **Ranges & Views**: Favor the `<ranges>` library for data transformation over imperative `for` loops. Use functional pipelines like `std::views::filter` and `std::views::transform` (map).
*   **Concepts**: Use standard concepts (`std::integral`, `std::floating_point`) or define custom constraints to restrict template parameters instead of legacy SFINAE or raw `typename`.
*   **Designated Initializers**: Use designated initializers for aggregates to increase readability when initializing structs.

### Modern C++ Style Guardrails
*   Avoid raw pointers (`new`/`delete`). Use smart pointers (`std::unique_ptr`, `std::shared_ptr`) or value semantics.
*   Prefer immutable variables by default (`const` or `constexpr`).
*   Leverage structured binding (`auto [x, y] = ...`) for unpacking tuples, pairs, or structs.

#### Example: Functional Mapping and Filtering (C++20)
```cpp
#include <iostream>
#include <vector>
#include <ranges>
#include <numeric>

struct TrackedObject {
    int id;
    double confidence;
    bool active;
};

double get_active_average_confidence(const std::vector<TrackedObject>& objects) {
    // Functional pipeline: filter active -> map to confidence
    auto active_confidences = objects 
        | std::views::filter([](const auto& obj) { return obj.active; })
        | std::views::transform([](const auto& obj) { return obj.confidence; });

    // Materialize or reduce
    double sum = 0.0;
    int count = 0;
    for (double conf : active_confidences) {
        sum += conf;
        count++;
    }
    
    return count == 0 ? 0.0 : sum / count;
}
