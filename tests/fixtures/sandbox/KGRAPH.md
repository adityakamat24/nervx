# sandbox
Python
_Graph: 76 symbols, 184 edges_

## Module Map
  agents/  3 classes, 14 functions
  game/  3 classes, 16 functions
  memory/  2 classes, 9 functions
  server/  1 classes, 4 functions
  tests/  tests

## Entry Points
  main.py::main() -> None

## Key Flows
  main_to_GameState: main → GameState
  main_to_EventBus_get_instance: main → EventBus.get_instance
  domain_agent.py: base_agent.py → claude_agent.py → gpt_agent.py → test_agent.py
  domain_agent: BaseAgent → ClaudeAgent → GPTAgent → test_create_agent → test_agent_decide → test_agent_repr → ...
  domain___init__: BaseAgent.__init__ → EventBus.__init__ → GameState.__init__ → VotingSystem.__init__ → MemoryIndex.__init__ → MemoryStore.__init__ → ...

## Detected Patterns
  EVENT_BUS: EventBus
    → Static call graph will NOT show all subscribers. 2 listener registrations found across 2 files. Search for .on()/.subscribe() calls.
  FACTORY: create_agent
    → Factory with 5 call sites. When adding new types, add a case here.
  STRATEGY: BaseAgent
    → BaseAgent is a strategy base with 2 implementations: ClaudeAgent, GPTAgent. When modifying the interface, update all implementations.
  REPOSITORY: GameState
    → Data access centralized in GameState. Don't bypass with direct storage calls.
  REPOSITORY: MemoryStore
    → Data access centralized in MemoryStore. Don't bypass with direct storage calls.
  SINGLETON: EventBus
    → EventBus is a singleton. Use the class method, don't instantiate directly.

## Fragile Zones
  memory/store.py::MemoryStore  importance=40.0, no tests
  main.py::main  importance=33.5, no tests
  memory/store.py::save  importance=31.5, no tests
  game/events.py::get_instance  importance=30.5, no tests, contract conflict
  game/events.py::emit  importance=20.5, no tests
