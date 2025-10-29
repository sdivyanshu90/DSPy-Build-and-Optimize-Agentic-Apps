# DSPy: Build and Optimize Agentic Apps

This repository contains my personal notes, code exercises, and project work for the course **[DSPy: Build and Optimize Agentic Apps](https://www.deeplearning.ai/short-courses/dspy-build-optimize-agentic-apps/)** by **[DeepLearning.AI](https://www.deeplearning.ai/)**, offered in collaboration with **[DataBricks](https://databricks.com/)**w.

-----

## About This Course

Agentic AI applications tackle complex tasks such as document automation, question-answering, and multi-step decision-making. However, building these applications can become complex, and one challenge is writing and maintaining good prompts. **DSPy** is a flexible open-source framework that simplifies your application’s interaction with LLMs. It streamlines your workflow by utilizing modular blocks in which you can provide a dataset of inputs and desired output, and systematically build, trace, and optimize your application.

This course teaches you how to use DSPy to build and optimize LLM-powered applications. You’ll write programs using DSPy’s signature-based programming model, debug them with MLflow tracing, and automatically improve their accuracy with DSPy Optimizer. Along the way, you’ll see how DSPy helps you easily switch models, manage complexity, and build agents that are both powerful and easy to maintain.

You’ll immediately put the concepts to work, first by coding a sentiment classifier in roughly 30 lines, then stretching the same pattern into a “Name the Celebrity” guessing game. Next, you’ll trace every step of a travel-booking assistant with MLflow, and wrap up by letting DSPy Optimizer lift a Wikipedia-based RAG agent’s accuracy, all without hand-tuning prompts.

**In detail, you’ll learn:**

  * What makes DSPy different from other development frameworks, and how its signature-based design enables flexible, model-agnostic development.
  * How to compose agentic apps by chaining DSPy modules like `Predict`, `ChainOfThought`, and `React`, and debug them using MLflow tracing, demonstrated in both a sentiment analyzer and a “Name the Celebrity” guessing game.
  * How to visualize and interpret your DSPy programs using MLflow, an open-source MLOps framework for observability that makes it easy to understand submodule behavior and catch issues early.
  * How to use DSPy Optimizer to automatically improve your program quality through prompt tuning and few-shot examples, seen through an example of optimizing a Wikipedia RAG agent that improves from 31% to 54% exact-match accuracy.

By the end of this course, you’ll be able to build structured, robust, and adaptable GenAI applications with DSPy, ready to run on whichever LLM comes next.

-----

## Course Topics

This course is structured into four main modules:

<details>
<summary><strong>1. Introduction to DSPy</strong></summary>

This foundational module introduces the core philosophy and necessity of a framework like **DSPy**. It starts by defining the landscape of modern "agentic" AI applications. These are not simple, one-shot "text-in, text-out" tasks; they are complex, multi-step processes designed to tackle sophisticated goals like automating document processing, building robust question-answering systems over private data, or creating assistants that can make decisions and take actions.

The module then pinpoints the central challenge in building these applications: **prompt engineering**. As applications grow in complexity, so do the prompts. They become long, brittle, and difficult to maintain. A small change in one part of a prompt can have unpredictable effects, and "prompt rot" becomes a significant engineering burden. Furthermore, prompts are often tightly coupled to a specific Large Language Model (LLM). If you want to switch from GPT-4 to Llama 3 or a new, more efficient model, you often have to rewrite and re-tune all your prompts from scratch.

This is where DSPy is introduced as a new paradigm. The key takeaway is understanding **what makes DSPy different** from other development frameworks (like LangChain or LlamaIndex, which often focus on pre-built components). DSPy is presented not as a collection of prompts, but as a *programming model* and *compiler* for language models. It provides a systematic way to build, debug, and optimize complex LLM pipelines.

The core concept introduced here is DSPy's **signature-based design**. This is a powerful abstraction that separates the *logic* of your program from the *prompts* used to execute it. Instead of writing a massive prompt, you define a simple "signature" for each step. A signature declares the inputs and outputs for a small, self-contained task (e.g., `input: context, question -> output: answer`). DSPy then *compiles* this signature, along with others in your pipeline, into an effective prompt for the target LLM. This design is what enables **flexible, model-agnostic development**. You write your program's logic once, and DSPy handles the complex task of "prompting" the model, allowing you to switch LLMs with minimal code changes. This module sets the stage by showing *why* a framework like DSPy is essential for moving from ad-hoc scripting to disciplined, maintainable, and powerful GenAI application development.

</details>

<details>
<summary><strong>2. DSPy Programming - Signatures and Modules</strong></summary>

This module is the practical, hands-on core of the course. It moves from the "why" of the introduction to the "how" of building. You learn to write programs using DSPy’s fundamental building blocks: **Signatures** and **Modules**.

**Signatures** are explored in depth. You learn the simple, declarative syntax for defining the inputs and outputs of any LLM-driven step. For example, a sentiment classification signature might be `sentence -> sentiment`, and a more complex RAG signature might be `context, question -> answer`. The power of this approach is its simplicity and expressiveness. You are not thinking about *how* to phrase the prompt; you are simply declaring the *data transformation* you want the LLM to perform.

**Modules** are the second key component. Modules are the "verbs" of DSPy—they are the components that *do* the work. You learn how to use pre-built modules and wrap your signatures inside them. The course covers the most important modules:

  * **`dspy.Predict`**: The simplest module, used for "zero-shot" tasks. You give it a signature, and it executes it.
  * **`dspy.ChainOfThought`**: This module implements the "Chain of Thought" reasoning technique. By simply changing `dspy.Predict` to `dspy.ChainOfThought` (and adding `reasoning` to your signature), you instruct the LLM to "think step-by-step" before giving its final answer, which dramatically improves performance on complex tasks.
  * **`dspy.ReAct`**: This module implements the "ReAct" (Reasoning + Action) agentic framework. This is used for building true agents that can interact with external tools (like a search engine or a calculator) to gather information before answering.

The key lesson is learning how to **compose agentic apps** by chaining these modules together. You don't build one giant module; you build a pipeline of small, specialized modules. Your RAG system, for instance, would be a pipeline: one module retrieves relevant documents, and a second module (using a `context, question -> answer` signature) generates the answer based on those documents.

To make these concepts concrete, the course walks you through building two applications from scratch. First, you build a **sentiment classifier in roughly 30 lines of code**, demonstrating the remarkable efficiency of the DSPy model. Second, you expand on this pattern to create a more complex, multi-step **“Name the Celebrity” guessing game**, which requires chaining multiple modules to manage state and logic, solidifying your understanding of how to build true agentic applications.

</details>

<details>
<summary><strong>3. Debug Your DSPy Agent with MLflow Tracing</strong></summary>

Once you build a multi-step agent, a new problem arises: a lack of transparency. If your 5-step RAG agent gives a bad answer, where did it go wrong? Was it the retrieval step (bad context)? Was it the generation step (hallucination)? Without a proper tool, the pipeline is a "black box," and debugging is reduced to guesswork.

This module provides the solution: **MLflow Tracing**. It introduces **MLflow** as an open-source MLOps framework designed for end-to-end machine learning lifecycle management, with a powerful focus on **observability**. The course demonstrates how DSPy's integration with MLflow makes it trivial to visualize, interpret, and debug your programs.

You learn how to instrument your DSPy applications so that every call to an LLM, every input, and every output is logged and visualized. This "tracing" allows you to see the entire flow of execution for any given request. You can see the exact prompt that was *actually* sent to the LLM (the one DSPy compiled), the LLM's raw response, and how that response was parsed and passed to the next module in the chain.

The module emphasizes how this tracing makes it **easy to understand submodule behavior and catch issues early**. Instead of just knowing the *final* answer was wrong, you can pinpoint the *exact* submodule that failed. You might see that your `ChainOfThought` module's `reasoning` was flawed, or that your `ReAct` agent failed to use a tool correctly.

To demonstrate this, the course uses a practical example: a **travel-booking assistant**. This is a perfect case study because it involves multiple, dependent steps (e.g., finding flights, then booking a hotel, then confirming a car). You learn to trace every step of this assistant's "conversation" with the user, visually identifying bottlenecks, errors, and areas for improvement. This module empowers you to move from building "black box" agents to building transparent, production-ready systems that you can confidently debug and maintain.

</details>

<details>
<summary><strong>4. Optimizing Agents with DSPy Optimizer</strong></summary>

This final module introduces the most powerful and unique feature of the DSPy framework: the **DSPy Optimizer**. This is what truly sets DSPy apart, transforming it from a mere programming model into a full-fledged compiler.

The problem is this: even with a well-structured pipeline of signatures and modules, your agent's performance is still dependent on the quality of the prompts and the few-shot examples provided to the LLM. Traditionally, improving this performance requires a painful, manual process of "prompt tuning" or "hand-tuning"—tweaking the wording of prompts, manually writing 5-10 good examples, running tests, and repeating the cycle for hours. This is brittle, time-consuming, and not scalable.

The DSPy Optimizer automates this entire process. You learn how to use the optimizer to **automatically improve your program quality**. You provide the optimizer with three things:

1.  Your DSPy program (the pipeline of modules).
2.  A small dataset of training examples (input/output pairs).
3.  A metric to optimize for (e.g., exact match, "is the answer factually correct?").

The optimizer then "runs" your program, tries different variations of prompts and few-shot examples, and systematically figures out the best possible "instructions" for each module in your pipeline to maximize the metric you defined. It effectively *learns* the optimal prompt for you, **all without you hand-tuning a single prompt**. It uses techniques like **prompt tuning** and **automatic few-shot example generation** to find the combination that works best for your specific task and your chosen LLM.

The results are dramatic. The course provides a stunning, real-world example of optimizing a **Wikipedia-based RAG agent**. You see the agent's initial "zero-shot" performance, which might only achieve **31% exact-match accuracy**. This is a poor-quality, unreliable agent. Then, you watch as the DSPy Optimizer is run on this *exact same program*. After the optimization process, the *same* program's accuracy is lifted to **54% exact-match accuracy**. This is not a small tweak; it's a transformative improvement in quality, achieved completely automatically. This module proves the core promise of DSPy: you focus on the *logic* (the signatures and modules), and the optimizer handles the *performance* (the prompts and tuning).

</details>

-----

## Acknowledgement

This repository is for my personal learning and development. All course materials, lectures, and projects are the intellectual property of **DeepLearning.AI** and **DataBricks**.

All rights are held by DeepLearning.AI. This repository is for educational and non-commercial purposes only.