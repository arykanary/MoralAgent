"""Nodes of the moral-principles knowledge graph.

Each node is an internationally recognized ethical/moral norm for AI,
extracted from guidelines published by governments, institutions, and
academic bodies (e.g. UNESCO, OECD, EU HLEG, ACM). Nodes carry a tier
('basic' principle or 'guiding' principle for special situations / specific
groups) and 'related' edges to neighbor principles.
"""

PRINCIPLES: dict[str, dict] = {
    "fairness": {
        "tier": "basic",
        "description": (
            "Treat all people equitably and without discrimination; give everyone "
            "equal consideration unless a morally relevant reason justifies a difference."
        ),
        "keywords": ["fair", "fairness", "equality", "equal", "discrimination", "bias", "justice"],
        "related": ["justice", "respect for others", "cooperating with others"],
    },
    "justice": {
        "tier": "basic",
        "description": (
            "Distribute benefits and burdens in society according to what is deserved and "
            "procedurally fair; do not let advantages accrue to some at the unfair expense of others."
        ),
        "keywords": ["justice", "deserve", "deserved", "merit", "equity", "fair share", "order"],
        "related": ["fairness", "respect for others"],
    },
    "peace": {
        "tier": "basic",
        "description": (
            "Avoid escalating conflict, hostility, or harm between people; prefer resolutions "
            "that maintain peaceful coexistence."
        ),
        "keywords": ["peace", "peaceful", "conflict", "harmony", "violence", "rage", "argue"],
        "related": ["safety", "cooperating with others"],
    },
    "freedom": {
        "tier": "basic",
        "description": (
            "Respect the autonomy and liberty of individuals to make their own choices, "
            "unless exercising that freedom unjustifiably harms others."
        ),
        "keywords": ["freedom", "autonomy", "liberty", "choice", "consent", "own decision", "permission"],
        "related": ["respect for others", "safety"],
    },
    "cooperating with others": {
        "tier": "basic",
        "description": (
            "Support and do not undermine collective activities and shared institutions that "
            "people rely on (e.g. queues, shared property, agreements)."
        ),
        "keywords": ["cooperate", "cooperation", "shared", "collective", "queue", "agreement", "rule", "norm"],
        "related": ["fairness", "justice", "respect for others"],
    },
    "respect for others": {
        "tier": "basic",
        "description": (
            "Treat other people with dignity and consideration; do not treat them merely as "
            "obstacles or instruments to one's own ends."
        ),
        "keywords": ["respect", "dignity", "consideration", "polite", "rude", "impolite", "insult"],
        "related": ["fairness", "freedom"],
    },
    "safety": {
        "tier": "basic",
        "description": (
            "Avoid creating risks of physical or psychological harm to people, including "
            "indirect and long-term risks."
        ),
        "keywords": ["safety", "safe", "harm", "danger", "dangerous", "risk", "injury", "hurt"],
        "related": ["peace", "respect for vulnerable groups"],
    },
    "respect for vulnerable groups": {
        "tier": "guiding",
        "description": (
            "Special care is owed to children, the elderly, the sick, people with "
            "disabilities, and anyone in a position of reduced power or dependence."
        ),
        "keywords": ["child", "children", "elderly", "sick", "disabled", "vulnerable", "weak", "patient"],
        "related": ["respect for others", "fairness"],
    },
    "avoid damage to property": {
        "tier": "guiding",
        "description": (
            "Do not damage, destroy, or misuse property that belongs to others; possession "
            "and use require authorization or an overriding justified reason."
        ),
        "keywords": ["property", "belongings", "damage", "destroy", "break", "borrow", "owner", "possessions"],
        "related": ["respect for others", "cooperating with others", "fairness"],
    },
    "honesty": {
        "tier": "guiding",
        "description": (
            "Communicate truthfully and keep commitments; deception and betrayal of trust "
            "undermine fair cooperation."
        ),
        "keywords": ["honest", "honesty", "lie", "deceive", "promise", "trust", "cheat"],
        "related": ["respect for others", "cooperating with others"],
    },
    "proportionality": {
        "tier": "guiding",
        "description": (
            "Responses and interventions should be proportionate to the situation; in "
            "emergencies or exceptional circumstances, deviating from a normal rule may be "
            "justified when the deviation is proportionate to the need."
        ),
        "keywords": ["proportion", "proportionate", "emergency", "exception", "urgent", "necessity", "justified"],
        "related": ["justice", "fairness", "safety"],
    },
}
