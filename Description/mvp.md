# Project Brief: AI Narrative Continuity Companion (MVP Specification)

This document establishes the technical blueprint for developing the Minimum Viable Product (MVP) of the AI Narrative Continuity Companion, as outlined in the **IBM Solution Design NarrativeAI.pdf**. The target of this MVP is to build a scalable, text-driven "Narrative State Engine" that resolves viewer confusion during complex storytelling without risking accidental plot spoilers.

---

## 1. MVP Core Scope & Constraints

To ensure a rapid and highly stable delivery, the MVP deliberately narrows down processing inputs and outputs to text-only operations:

* **Inputs:** Raw subtitle files (`.srt` or `.vtt`) and user-submitted text or voice questions. No raw video or image processing will be conducted in this phase.
* **Outputs:** Clean, context-grounded textual answers delivered via a localized user interface.
* **Dataset:** Restricted to one complete, complex test episode to establish absolute proof-of-concept.

---

## 2. Memory Architecture & The "1-Minute" Data Strategy

To prevent context window explosion and eliminate system lag, the data ingestion pipeline splits the episode text into precise temporal blocks:

### Chronological Chunking

* The raw subtitle file is parsed and grouped into static **1-minute chronological chunks**.
* Each chunk is assigned strict metadata tags: `[episode_number]`, `[start_timestamp]`, and `[end_timestamp]`.

### The Dynamic RAG Pipeline

* Instead of passing a giant, cumulative summary of the entire episode to the AI, the data chunks remain independent in the database.
* When a question is asked, the system fetches a localized "retrieval budget" containing only the immediate scene dialogue (the last 60 seconds) combined with the **Top 3 or 4 historically relevant chunks** pulled via semantic search.

---

## 3. The Spoiler Shield Layer (Data-Level Governance)

The core value proposition of this application is a strict, bulletproof **Spoiler Shield**. For the MVP, this layer is handled entirely via **database query isolation** rather than predictive machine learning logic:

```
[User Pauses Video at 00:24:30] 
              │
              ▼
[Execute DB Query Filter] ──► Drop all chunks where end_timestamp > 00:24:30
              │
              ▼
[Safe Context Window Pool] ──► Only contains events from 00:00:00 to 00:24:30
              │
              ▼
[Semantic Vector Search] ──► Fetch top matches based purely on user query

```

By enforcing a hard database boundary at the code level, future plot data is physically omitted before it ever reaches the LLM context window. The model cannot accidentally reveal a spoiler because it is completely blind to future text chunks.

---

## 4. IBM Cloud Service Architecture (7 Core AI/ML Services)

The system relies exclusively on the following seven integrated IBM Watsonx services to drive the multi-agent orchestration pipeline:

### 1. watsonx.ai Studio (`watsonx-Hackathon WS`)

* **Role:** Model Development & Prompt Engineering Environment.
* **Application:** Used to build, evaluate, and test the parsing prompts that transform raw subtitle timelines into clean, semantic text blocks during data pre-processing.

### 2. watsonx.ai Runtime (`watsonx-Hackathon WML`)

* **Role:** High-Performance LLM Deployment Layer.
* **Application:** Powers the live generation engine that hosts the foundational model, executing real-time narrative inference when answering user questions.

### 3. Natural Language Understanding (`watsonx-Hackathon NLU`)

* **Role:** Entity Extraction & Metadata Tagging.
* **Application:** Scans incoming subtitle chunks to automatically extract key entities (character names, locations, political factions) to enrich the vector indexing for more accurate semantic search.

### 4. watsonx Orchestrate (`watsonx-Hackathon Orchestrate`)

* **Role:** Core Multi-Agent Backbone.
* **Application:** Coordinates the operational flow. It intercepts the user pause trigger, grabs the active timestamp, runs the database filter, passes data to the LLM, and surfaces the output to the UI.

### 5. watsonx.governance (`watsonx-Hackathon GOV`)

* **Role:** Safety, Bias, and Compliance Monitoring.
* **Application:** Audits model responses for accuracy, tracks latency, checks for potential hallucinations, and ensures that the data boundaries set by the Spoiler Shield are strictly respected.

### 6. Speech to Text (`watsonx-Hackathon STT`)

* **Role:** Multimodal Input Facilitator.
* **Application:** Converts voice-activated user queries (e.g., verbal questions asked to a second-screen companion or remote control) into clean text strings for the processing engine.

### 7. Text to Speech (`watsonx-Hackathon TTS`)

* **Role:** Multimodal Output Delivery.
* **Application:** Converts the generated text response into clean audio, enabling the application to optionally read summaries out loud to sustain a seamless "lean-back" viewing environment.

---

## 5. Blueprint for Post-MVP Modularity (Future-Proofing)

To make sure we do not have to rebuild or discard any code after the hackathon MVP, the system enforces a strict **Separation of Concerns**. The architecture decouples data parsing from the central reasoning layer:

* **Agnostic Data Ingestion:** The database schema treats all narrative knowledge exactly the same, requiring only a text payload and a strict timestamp tag.
* **Scaling Data Sources:** When we scale past the MVP to incorporate Fan Wikis, Fandom lore, or detailed character family trees, these sources will be broken down and tagged into specific episode timestamps. They will flow through the exact same query filtering code without requiring rewrites.
* **Scaling Inputs (Computer Vision):** When visual models are added later to analyze video frames (detecting facial recognition or objects on screen), the output will simply convert into text-based metadata tags (e.g., `[Character X is on screen]`). This text will feed directly into the existing Perception Agent interface, keeping the backend engine fully intact.