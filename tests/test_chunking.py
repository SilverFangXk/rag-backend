from app.chunking import build_chunks, fmt_date, fmt_month_long, split_text

DATA = {
    "projects": [
        {"slug": "etl-pipeline", "title": "ETL Pipeline", "summary": "Spark + Kafka pipeline", "tags": ["Spark", "Kafka"],
         "githubUrl": "https://github.com/x/etl",
         "steps": [{"label": "Ingest", "title": "Ingestion", "text": "Kafka topics feed Spark Streaming."}]},
        {"slug": "hidden", "title": "Secret", "visible": False},
    ],
    "skills": [
        {"name": "Python", "category": "Data Engineering"},
        {"name": "Spark", "category": "Data Engineering"},
        {"name": "Docker", "category": "Cloud & DevOps"},
    ],
    "experience": [
        {"id": "e1", "role": "Data Intern", "company": "Nexteer", "startDate": "2026-07-01", "endDate": "",
         "description": "**Built** dashboards.\n\nAutomated reports.", "projectSlugs": ["etl-pipeline"]},
    ],
    "education": [{"id": "ed1", "degree": "Engineering Cycle", "school": "ENSA Fès", "startYear": "2024", "endYear": "2027"}],
    "certifications": [{"id": "c1", "name": "Azure Data Fundamentals", "issuer": "Microsoft", "date": "Mar 2026"}],
    "settings": {
        "site": {"heroTitle": "Data & AI Engineer", "cvUrl": "https://cv.example/fr.pdf"},
        "next": {"seeking": "PFE Internship", "availableFrom": "2027-01", "focusAreas": ["Big Data", "AI"]},
        "contact": {"email": "a@b.c", "linkedin": "khadija-chkhichkh"},
    },
}


def by_id(chunks):
    return {c.id: c for c in chunks}


def test_unique_ids_and_hidden_skipped():
    chunks = build_chunks(DATA, "Khadija C")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert not any("Secret" in c.text for c in chunks)


def test_skills_grouped_by_category():
    c = by_id(build_chunks(DATA, "K"))
    assert "Python, Spark" in c["skills--data-engineering"].text
    assert "Docker" in c["skills--cloud-devops"].text


def test_experience_dates_markup_and_related_projects():
    text = next(c.text for c in build_chunks(DATA, "K") if c.id.startswith("experience--"))
    assert "Jul 2026 - Present" in text
    assert "**" not in text and "Built dashboards" in text
    assert "ETL Pipeline" in text  # slug resolved to the project title


def test_project_steps_carry_the_project_title():
    c = by_id(build_chunks(DATA, "K"))
    step = c["projects--etl-pipeline--step1--0"]
    assert step.text.startswith("Project: ETL Pipeline - Ingestion")


def test_profile_availability_contact():
    c = by_id(build_chunks(DATA, "K"))
    assert "January 2027" in c["profile--availability"].text
    assert "https://linkedin.com/in/khadija-chkhichkh" in c["profile--contact"].text
    assert "cv.example/fr.pdf" in c["profile--main"].text


def test_extra_facts_sections_and_comments_ignored():
    md = "<!-- hidden -->\n# Title\n\n## Languages\nArabic, French.\n\n## Empty\n"
    c = [x for x in build_chunks({"settings": {}}, "K", md) if x.section == "About"]
    assert len(c) == 1 and "Arabic" in c[0].text and "hidden" not in c[0].text


def test_split_text_respects_limit_and_loses_nothing():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    parts = split_text(text, 300)
    assert all(len(p) <= 300 for p in parts)
    assert len(" ".join(parts).split()) == len(text.split())


def test_date_helpers():
    assert fmt_date("2026-07-15") == "Jul 2026"
    assert fmt_date("Summer 2026") == "Summer 2026"
    assert fmt_month_long("2027-01") == "January 2027"


def test_fingerprint_changes_with_text_and_salt():
    a = build_chunks(DATA, "K")[0]
    assert a.fingerprint("m:768") != a.fingerprint("m:512")
    assert a.fingerprint("x") == a.fingerprint("x")
