Jupiter Notebook Unsloth for training and evaluating the model for generating valid diagrams using Mermaid syntax.
 

Once the PyTorch model generates valid syntax, it must be converted for edge deployment.

1. Merge Weights: Merge the trained LoRA adapters back into the base model weights to create a standalone PyTorch model.

2. Conversion: Use the Google AI Edge Torch framework (ai-edge-torch) or the Keras KerasNLP export utilities to convert the PyTorch artifact into the .tflite format compatible with the LiteRT runtime.

3. Quantization: Apply post-training quantization (e.g., INT8 dynamic range quantization) during the conversion step to ensure the model fits within mobile memory constraints.

The notebook will run on a Google Colab T4 GPU (16GB VRAM). Optimize for strict memory constraints using 4-bit NormalFloat (NF4) quantization and a LoRA rank of 16.
Model: `https://huggingface.co/google/gemma-2-2b-it`
Example Notebook: `https://github.com/google-gemma/cookbook/blob/main/FunctionGemma/%5BFunctionGemma%5DFinetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb`
 
Dataset: `https://huggingface.co/datasets/colinfrisch/diagrams-mermaid-filtered`
 
For teaching rigid syntax like D3 or Mermaid, configure the LoRA adapter to target all linear modules (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj) rather than just attention heads. This maximizes the model's capacity to memorize new syntactic structures.
 
Standard cross-entropy loss validation is insufficient for code generation.

1. Automated Parsing: Implement a custom validation script that extracts the generated code blocks during evaluation.

2. Compilation Check: Pass Mermaid outputs through the mmdc (Mermaid CLI) and D3 outputs through a headless JavaScript parser to verify syntactic validity. Track the compilation success rate rather than just token accuracy.

 
To use this dataset for fine-tuning a Gemma model using the QLoRA pipeline, you must parse the stringified conversation format and map it into Gemma's specific instruction template (<start_of_turn>user...).
 
<dataset_example_csv>
messages
"[{'role': user, 'content': 'Multiple Intelligences, Circular Diagram, elements: Circle divided into eight segments, each labeled with different types of intelligences: Musical, Bodily-Kinesthetic, Interpersonal, Verbal-Linguistic, Logical-Mathematical, Naturalist, Self Smart, and Picture Smart, with \'Multiple Intelligences\' in the center.'}, {'role': assistant, 'content': '```mermaid
mindmap
  root((Multiple Intelligences))
    Musical
    Bodily-Kinesthetic
    Interpersonal
    Verbal-Linguistic
    Logical-Mathematical
    Naturalist
    Self Smart
    Picture Smart
```'}]"
"[{'role': user, 'content': 'Iterative Product Development, Circular Flow Diagram, elements: A circular design with seven numbered elements (1 to 7) representing steps in iterative product development, each accompanied by descriptive text including \'Continuous improvement\', \'Collaboration and cross-functional teams\', \'Data-driven decision making\', \'Rapid prototyping\', \'Agile project management\', and \'Minimum viable products (MVPs)\'.'}, {'role': assistant, 'content': '```mermaid
flowchart LR
  %% Iterative Product Development (circular flow)
  subgraph LOOP[Iterative Product Development]
    direction LR
    n1((""01\\nContinuous feedback loops""))
    n2((""02\\nMinimum viable products (MVPs)""))
    n3((""03\\nAgile project management""))
    n4((""04\\nRapid prototyping""))
    n5((""05\\nData-driven decision making""))
    n6((""06\\nCollaboration and cross-functional teams""))
    n7((""07\\nContinuous improvement""))
    n1 --> n2 --> n3 --> n4 --> n5 --> n6 --> n7 --> n1
  end
  classDef step fill:#ffffff,stroke:#666,stroke-width:2px,color:#111;
  class n1,n2,n3,n4,n5,n6,n7 step;
```'}]"
"[{'role': user, 'content': 'ADDIE Model of Instructional Design, Flow Diagram, elements: Five sequential blocks labeled ADDIE (Analysis, Design, Development, Implementation, Evaluation) with brief descriptions below each block, connected by arrows indicating the flow of the process.'}, {'role': assistant, 'content': '```mermaid
flowchart LR
  A[""A<br/>Analysis<br/><span style=\'font-size:12px\'>Identify the instructional problem, establish goals and objectives, and analyze learners and context.</span>""] -->
  D1[""D<br/>Design<br/><span style=\'font-size:12px\'>Plan the learning objectives, assessment instruments, content, and instructional strategy.</span>""] -->
  D2[""D<br/>Development<br/><span style=\'font-size:12px\'>Create and assemble the learning materials and resources; draft, test, and revise.</span>""] -->
  I[""I<br/>Implementation<br/><span style=\'font-size:12px\'>Deliver or distribute the instruction; train facilitators and prepare learners.</span>""] -->
  E[""E<br/>Evaluation<br/><span style=\'font-size:12px\'>Assess effectiveness and quality (formative and summative) and refine the instruction.</span>""]
  classDef a fill:#4db6ff,stroke:#1b5e80,color:#fff,rx:6,ry:6;
  classDef d fill:#7ee3c6,stroke:#1b5e20,color:#083b2a,rx:6,ry:6;
  classDef dev fill:#ff8a65,stroke:#8d3c2f,color:#fff,rx:6,ry:6;
  classDef i fill:#ff6b6b,stroke:#7a1f1f,color:#fff,rx:6,ry:6;
  classDef e fill:#26c6da,stroke:#006064,color:#fff,rx:6,ry:6;
  class A a;
  class D1 d;
  class D2 dev;
  class I i;
  class E e;
```'}]"
"[{'role': user, 'content': 'Bloom\'s Taxonomy, Pyramid Diagram, elements: Pyramid structure divided into six layers labeled: Knowledge, Comprehension, Application, Analysis, Synthesis, Evaluation, with increasing complexity from bottom to top.'}, {'role': assistant, 'content': '```mermaid
flowchart TB
  K[""Knowledge""] --> C[""Comprehension""] --> A[""Application""] --> AN[""Analysis""] --> S[""Synthesis""] --> E[""Evaluation""]
  classDef k fill:#F6C58B,stroke:#000,stroke-width:1px,color:#333;
  classDef c fill:#F7E7A3,stroke:#000,stroke-width:1px,color:#333;
  classDef a fill:#BFDFAE,stroke:#000,stroke-width:1px,color:#333;
  classDef an fill:#A9CFF2,stroke:#000,stroke-width:1px,color:#333;
  classDef s fill:#CFC7F2,stroke:#000,stroke-width:1px,color:#333;
  classDef e fill:#E1C6D0,stroke:#000,stroke-width:1px,color:#333;
  class K k;
  class C c;
  class A a;
  class AN an;
  class S s;
  class E e;
```'}]"
"[{'role': user, 'content': 'Bloom\'s Taxonomy, Pyramid Diagram, elements: Pyramid structure with six layers labeled: Remember, Understand, Apply, Analyze, Evaluate, and Create; a key on the right indicates College and High School levels.'}, {'role': assistant, 'content': '```mermaid
flowchart TB
  %% Bloom\'s Taxonomy Pyramid
  A[Create]
  B[Evaluate]
  C[Analyze]
  D[Apply]
  E[Understand]
  F[Remember]
  A --> B --> C --> D --> E --> F
  classDef top fill:#f6b11a,stroke:#ffffff,color:#000,font-weight:bold;
  classDef lvl2 fill:#c8f26a,stroke:#ffffff,color:#000,font-weight:bold;
  classDef lvl3 fill:#43e04b,stroke:#ffffff,color:#000,font-weight:bold;
  classDef lvl4 fill:#38d88a,stroke:#ffffff,color:#000,font-weight:bold;
  classDef lvl5 fill:#43cbd8,stroke:#ffffff,color:#000,font-weight:bold;
  classDef base fill:#3e6fb8,stroke:#ffffff,color:#000,font-weight:bold;
  class A top;
  class B lvl2;
  class C lvl3;
  class D lvl4;
  class E lvl5;
  class F base;
  %% Right-side key
  subgraph KEY[ ]
    direction TB
    K1[College<br/>Level]
    K2[High<br/>School<br/>Level]
  end
  K1 -.-> A
  K1 -.-> B
  K1 -.-> C
  K2 -.-> D
  K2 -.-> E
  K2 -.-> F
  classDef key fill:#ffffff,stroke:#ffffff,color:#000;
  class K1,K2 key;
  linkStyle 0,1,2,3,4 stroke-width:0px;
  linkStyle 5,6,7,8,9,10 stroke:#999,stroke-dasharray:6 4,stroke-width:2px;
  style KEY fill:transparent,stroke:transparent;
```'}]"
"[{'role': user, 'content': 'Zone of Proximal Development in Teaching, Graph, elements: Diagonal axis labeled \'Level of Challenge\' and \'Level of Competence\' with \'Zone of Proximal Development\' highlighted. Areas labeled: \'Anxiety\' above, \'Boredom\' below, indicating learner independence and assistance.'}, {'role': assistant, 'content': '```mermaid
flowchart LR
  %% Axes
  subgraph Chart[""Zone of Proximal Development in Teaching""]
    direction LR
    X0(("" "")) --- X1[""Level of competence →""]
    Y0(("" "")) --- Y1[""↑ Level of challenge""]
    %% Regions
    Anxiety[""Anxiety<br/>(too challenging without support)""]
    ZPD[""Zone of Proximal Development<br/>(with assistance / scaffolding)""]
    Boredom[""Boredom<br/>(too easy / already mastered)""]
    %% Place regions conceptually
    Y1 --> Anxiety
    X1 --> Boredom
    Anxiety --> ZPD --> Boredom
    %% Notes
    N1[""What the learner will be able to achieve independently (later)""]
    N2[""What the learner can currently achieve independently""]
    N3[""What the learner can achieve with assistance""]
    N4[""Scaffolding occurs through the support of the more knowing other""]
    Anxiety -.-> N1
    ZPD -.-> N3
    Boredom -.-> N2
    ZPD -.-> N4
  end
  %% Styling
  style ZPD fill:#9ad86b,stroke:#2f6b2f,stroke-width:2px
  style Anxiety fill:#ffffff,stroke:#999,stroke-width:1px
  style Boredom fill:#ffffff,stroke:#999,stroke-width:1px
```'}]"
"[{'role': user, 'content': 'Big Five Personality Traits, Circular Diagram, elements: Five colored circles representing personality traits: Neuroticism, Extraversion, Openness, Agreeableness, and Conscientiousness, each with descriptors for low and high levels labeled around the circles.'}, {'role': assistant, 'content': '```mermaid
flowchart TB
  %% Big Five Personality Traits - Circular Diagram (approximated)
  classDef trait fill:#ffffff,stroke:#333,stroke-width:1px,color:#111,font-weight:bold;
  classDef neuro fill:#d81b1b,stroke:#333,stroke-width:1px,color:#fff;
  classDef extra fill:#ff8f00,stroke:#333,stroke-width:1px,color:#111;
  classDef open fill:#ffd54f,stroke:#333,stroke-width:1px,color:#111;
  classDef agree fill:#8bc34a,stroke:#333,stroke-width:1px,color:#111;
  classDef consc fill:#42a5f5,stroke:#333,stroke-width:1px,color:#111;
  classDef label fill:#ffffff,stroke:transparent,color:#222,font-size:12px;
  subgraph S[""Big Five Personality Traits""]
    direction LR
    N((NEUROTICISM)):::neuro
    E((EXTRAVERSION)):::extra
    O((OPENNESS)):::open
    A((AGREEABLENESS)):::agree
    C((CONSCIENTIOUSNESS)):::consc
    %% Arrange traits in a line to mimic circular layout row
    N --- E --- O --- A --- C
    %% Neuroticism descriptors
    Nhi[""high<br/>depressivity<br/>emotional lability<br/>shamefulness""]:::label
    Nlo[""low<br/>fearlessness<br/>shamelessness""]:::label
    Nhi -.-> N
    Nlo -.-> N
    %% Extraversion descriptors
    Ehi[""high<br/>excitement seeking<br/>attention seeking""]:::label
    Elo[""low<br/>social withdrawal<br/>detached coldness""]:::label
    Ehi -.-> E
    Elo -.-> E
    %% Openness descriptors
    Ohi[""high<br/>magical thinking<br/>eccentricity""]:::label
    Olo[""low<br/>inflexible<br/>close-minded""]:::label
    Ohi -.-> O
    Olo -.-> O
    %% Agreeableness descriptors
    Ahi[""high<br/>submissiveness<br/>selflessness<br/>gullibility""]:::label
    Alo[""low<br/>deceitfulness<br/>manipulativeness<br/>callousness""]:::label
    Ahi -.-> A
    Alo -.-> A
    %% Conscientiousness descriptors
    Chi[""high<br/>perfectionism<br/>workaholism""]:::label
    Clo[""low<br/>distractiveness<br/>irresponsibility<br/>rashness""]:::label
    Chi -.-> C
    Clo -.-> C
  end
```'}]"
"[{'role': user, 'content': 'Big Five Personality Traits, Horizontal Diagram, elements: Five colored circles labeled O, C, E, A, N with corresponding traits: Openness - prefers routine; Conscientious - impulsive vs. disciplined; Extraversion - reserved vs. sociable; Agreeable - suspicious vs. trusting; Neuroticism - calm vs. anxious.'}, {'role': assistant, 'content': '```mermaid
flowchart LR
  %% Big Five Personality Traits — Horizontal Diagram
  classDef o fill:#7B61FF,stroke:#7B61FF,color:#ffffff,font-weight:bold;
  classDef c fill:#39B9FF,stroke:#39B9FF,color:#ffffff,font-weight:bold;
  classDef e fill:#31C48D,stroke:#31C48D,color:#ffffff,font-weight:bold;
  classDef a fill:#FF9E80,stroke:#FF9E80,color:#ffffff,font-weight:bold;
  classDef n fill:#D96CFF,stroke:#D96CFF,color:#ffffff,font-weight:bold;
  subgraph T[Big Five Personality Traits]
    direction LR
    O((O)):::o
    C((C)):::c
    E((E)):::e
    A((A)):::a
    N((N)):::n
    Otxt[""**Openness**<br/>prefers routine""]:::label
    Ctxt[""**Conscientious**<br/>impulsive vs. disciplined""]:::label
    Etxt[""**Extraversion**<br/>reserved vs. sociable""]:::label
    Atxt[""**Agreeable**<br/>suspicious vs. trusting""]:::label
    Ntxt[""**Neuroticism**<br/>calm vs. anxious""]:::label
    O --- Otxt
    C --- Ctxt
    E --- Etxt
    A --- Atxt
    N --- Ntxt
  end
  classDef label fill:transparent,stroke:transparent,color:#111827;
```'}]"
"[{'role': user, 'content': 'Bloom\'s Taxonomy, Pyramid Diagram, elements: Pyramid structure with layers labeled: Knowledge (bottom), Comprehension, Application, Analysis, Synthesis, and Evaluation (top), with color coding for each level.'}, {'role': assistant, 'content': '```mermaid
flowchart TB
  %% Bloom\'s Taxonomy Pyramid (stacked layers)
  subgraph BT[""Bloom\'s Taxonomy""]
    direction TB
    E[""Evaluation""]
    S[""Synthesis""]
    A[""Analysis""]
    Ap[""Application""]
    C[""Comprehension""]
    K[""Knowledge""]
    E --> S --> A --> Ap --> C --> K
  end
  %% Styling
  classDef eval fill:#fff36a,stroke:#333,color:#000;
  classDef synth fill:#ffa500,stroke:#333,color:#000;
  classDef analysis fill:#6a1b9a,stroke:#333,color:#fff;
  classDef app fill:#00a651,stroke:#333,color:#fff;
  classDef comp fill:#1e63c6,stroke:#333,color:#fff;
  classDef know fill:#e60000,stroke:#333,color:#fff;
  class E eval;
  class S synth;
  class A analysis;
  class Ap app;
  class C comp;
  class K know;
```'}]"
"[{'role': user, 'content': 'Iterative Scrum Process, Circular Flow Diagram, elements: Outer ring with five phases labeled: Planning/Prioritization, Requirements, Design/Analysis, Implementation/Testing, and Review/Retrospective. Central \'Scrum Cycle\' label. Top section showing \'Concept\' with four iterations labeled Scrum 1 to Scrum 4.'}, {'role': assistant, 'content': '```mermaid
flowchart TB
  %% Top iterative strip
  subgraph TOP[""Iterative Scrum Process""]
    direction LR
    C[Concept] --> S1[Scrum 1] --> S2[Scrum 2] --> S3[Scrum 3] --> S4[Scrum 4] --> CONT[Continue]
  end
  %% Circular Scrum cycle
  subgraph CYCLE[""Scrum Cycle""]
    direction LR
    P1[""1\\nPlanning/\\nPrioritization""] --> P2[""2\\nRequirements""] --> P3[""3\\nDesign/\\nAnalysis""] --> P4[""4\\nImplementation/\\nTesting""] --> P5[""5\\nReview/\\nRetrospective""] --> P1
  end
  %% Styling
  classDef topNode fill:#ffffff,stroke:#cfcfcf,stroke-width:2px,rx:10,ry:10,color:#0b6aa2,font-weight:700;
  class C,S1,S2,S3,S4,CONT topNode;
  classDef phase1 fill:#1e88e5,stroke:#ffffff,stroke-width:6px,color:#ffffff,font-weight:700;
  classDef phase2 fill:#f5a623,stroke:#ffffff,stroke-width:6px,color:#ffffff,font-weight:700;
  classDef phase3 fill:#4fc3d8,stroke:#ffffff,stroke-width:6px,color:#ffffff,font-weight:700;
  classDef phase4 fill:#7ed957,stroke:#ffffff,stroke-width:6px,color:#ffffff,font-weight:700;
  classDef phase5 fill:#7a7a7a,stroke:#ffffff,stroke-width:6px,color:#ffffff,font-weight:700;
  class P1 phase1;
  class P2 phase2;
  class P3 phase3;
  class P4 phase4;
  class P5 phase5;
```'}]"
</dataset_example_csv>
