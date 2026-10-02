# LLM Serving through Request Characterization and Policy Evaluation

## Background

Large Language Model (LLM) applications process workloads with significant variance in both computational demand and service requirements. A request may consist of a short prompt requiring a brief answer, or a long context producing an extensive generation. Consequently, this project treats an LLM request as more than a uniform API call: attributes such as task type, prompt length, expected output length, quality requirements, latency targets, privacy constraints, and budget all dictate how it should be served.

Similar heterogeneity exists on the serving infrastructure. Different inference backends and replicas vary in cost, processing capabilities, queue states, service times, memory pressure, latency profiles, failure rates, and availability.

A core architectural principle is the decoupling of decisions often coupled together. Selecting a logical model or provider determines the appropriate class of backend, whereas selecting a specific replica determines which instance executes the request. Decoupling these decisions enables independent analysis of each choice alongside their combined dynamics.

## Problem Statement

The objective is to design and evaluate a policy layer for heterogeneous LLM serving that translates incoming workload characteristics and real-time serving conditions into optimal routing actions. Rather than standardizing conventional gateway capabilities, this study investigates how different policy designs navigate trade-offs across cost, response quality, latency, reliability, and service-level objectives (SLOs).

The central question is:   
" How should an LLM-serving system characterize incoming demand and integrate it with dynamic backend conditions to make adaptive serving decisions under heterogeneous workloads?"

## Scope

- **Request Characterization:** Requests are specified by task type, prompt length, expected output length, required quality level, latency targets, privacy requirements, and cost constraints. These attributes serve as inputs to the decision layer to differentiate varied computational demands. 

- **Demand Estimation:** Recognizing that prompt and generation lengths drive inference load, synthetic and real evaluation workloads incorporate diverse combinations of prompt and generation lengths rather than treating requests uniformly. 

- **Multi-Objective Policy Layer:** The policy engine balances competing objectives (e.g., quality vs. cost and latency). A rule-based controller provides a baseline for interpretable decision-making without requiring complex reinforcement learning models initially. 

- **Decision Separation:** Model selection and replica selection are structured as distinct evaluation stages. Model selection specifies the target model class, while replica selection utilizes runtime metrics-such as queue depth, resource utilization, and recent latency-to choose an instance. 

- **Workload & Experimentation:** The experimental suite includes workload generators, configuration files, execution scripts, telemetry pipelines, and comparative visualization tools across bursty, heterogeneous, and fault-injected scenarios.

## Key Challenges(currently)

- **Representing Heterogeneous Demand:** Because prompt and output sizes vary widely, simple request counts fail to capture system load, requiring multi-dimensional demand representation.

- **Balancing Conflicting Objectives:** Trade-offs between cost, quality, and latency must be explicitly parametrized to support varying task priorities.

- **Coupling Between Decisions:** Model routing directly impacts replica load, while replica health influences model efficiency, requiring isolated and combined policy evaluations.

- **Dynamic System Conditions:** Fluctuations in queue length, memory pressure, and failure rates necessitate real-time adaptability beyond static lookup tables. 

- **Reproducible Evaluation under Resource Constraints:** Experiments are designed to be reproducible on limited hardware (single-GPU or CPU simulation) without sacrificing empirical rigor.

## Expected Output(but not limited to)

- **Software and Experimental Artifacts:** The project delivers a functional prototype and complete experimental artifact package rather than a new serving engine. The suite includes workload generators, policy configurations, failure scenarios, and automated analysis scripts.

- **Evaluation Framework:** The evaluation framework compares candidate policies-such as strongest-model, cheapest-model, static rule-based, and adaptive policies-against standard baselines like round-robin and least-loaded scheduling. Key metrics recorded across workload scenarios include TTFT, TPOT, throughput, p95/p99 latency, resource utilization, estimated cost, SLO attainment, and failure recovery dynamics.

- **Expected Outcome:**** The primary outcome is an empirically grounded decision framework identifying the specific workload and system conditions under which each adaptive serving policy optimizes cost, latency, quality, and SLO compliance.