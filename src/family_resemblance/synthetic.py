from __future__ import annotations

from .text import Document, normalize_token


DOMAIN_TEMPLATES: dict[str, list[str]] = {
    "auto_science": [
        "In a scientific article, the word {word} is used as an observable phenomenon that must be separated from neighboring processes and measured under different conditions.",
        "The researcher builds a model where {word} becomes a variable compared with a cause, an effect, and a hidden parameter.",
        "In a laboratory report, {word} describes not one essence, but a network of features that appear differently across experiments.",
        "When scientists debate the term {word}, they test which contexts produce stable results and which merely look similar from the outside.",
    ],
    "auto_law": [
        "In a legal document, {word} receives a procedural meaning: it matters who may invoke it and what consequences follow.",
        "The court treats {word} not as an everyday word, but as part of a norm linked to duties, evidence, and liability.",
        "In a contract, {word} can change meaning depending on the clause, the party, and the moment of performance.",
        "Legal practice shows that {word} rarely has one center: its meaning is refined through exceptions, precedents, and a balance of interests.",
    ],
    "auto_forum": [
        "On a forum, people use {word} freely: one participant describes personal experience, another debates rules, and a third recalls a similar story.",
        "In a discussion, {word} quickly changes shade because people connect it with work, family, money, and justice.",
        "Someone writes that {word} sounds too abstract until a concrete case from everyday life appears.",
        "In a chat, a dispute forms around {word}: some people demand a precise definition, while others understand it through examples.",
    ],
    "auto_fiction": [
        "In a novel, {word} appears at a moment of inward choice and sounds different from official speech.",
        "The author repeats {word} across scenes: first as hope, then as threat, and then as an almost forgotten promise.",
        "For the heroine, {word} has no dictionary definition because every close person fills it with a different pain.",
        "In literary text, {word} connects gesture, memory, and voice, but cannot be reduced to any single one of them.",
    ],
    "auto_politics": [
        "In political speech, {word} becomes a slogan, but different groups place incompatible expectations inside it.",
        "Public debate shows that {word} can mean reform, conflict, symbolic victory, or a method of mobilization.",
        "Activists use {word} as a sign of a shared goal, although their reasons and methods differ noticeably.",
        "In news analysis, {word} is tied to institutions, interests, and the struggle for the right to interpret events.",
    ],
    "auto_economy": [
        "In an economic report, {word} describes a resource, a risk, or an exchange mechanism that depends on market participants.",
        "For a company, {word} can be an advantage, a cost, or a constraint, depending on strategy and competitors.",
        "Analysts debate whether {word} is a measurable indicator or only a convenient metaphor for a complex situation.",
        "In business correspondence, {word} receives a practical meaning: deadlines, price, responsibility, and the expected result.",
    ],
    "auto_medicine": [
        "In a medical context, {word} is connected with the state of the body, symptoms, treatment, and patient response.",
        "The physician clarifies that {word} cannot be understood apart from history, age, lifestyle, and co-occurring factors.",
        "For the patient, {word} sounds like an experience; for the specialist, it becomes a set of signs, decisions, and probabilities.",
        "Clinical description shows that {word} changes meaning between complaint, diagnosis, and observation plan.",
    ],
    "auto_technology": [
        "In technical documentation, {word} names part of a system, interface behavior, or a condition under which the result changes.",
        "The developer discusses {word} through constraints, dependencies, failures, and validation methods on different data.",
        "In an engineering team, {word} can mean a user problem, an architectural decision, or a quality metric.",
        "The technical context makes {word} operational by linking it with inputs, outputs, and a repeatable procedure.",
    ],
}


def generate_synthetic_documents(targets: list[str] | tuple[str, ...]) -> list[Document]:
    documents: list[Document] = []
    for target in targets:
        word = normalize_token(target)
        if not word:
            continue

        for domain, templates in DOMAIN_TEMPLATES.items():
            text = "\n\n".join(template.format(word=word) for template in templates)
            documents.append(
                Document(
                    domain=domain,
                    source=f"synthetic://{word}/{domain}",
                    text=text,
                )
            )
    return documents
