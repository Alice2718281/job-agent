"""Simple validation checks for final hard filters and sponsorship extraction."""
from hard_filters import JobPreferenceFilter
from scorer import JobScorer


def main():
    scorer = object.__new__(JobScorer)
    hard_filter = JobPreferenceFilter()

    hybrid_sentence = (
        "This is a hybrid position and will require the person to work from our "
        "New York City office at minimum 3 days a week."
    )
    assert scorer._extract_sponsorship_evidence(hybrid_sentence) == "None found"

    passes, reason, evidence = hard_filter.passes_final_hard_filters({
        "title": "Product Analyst",
        "company": "Capital One",
        "location": "McLean, VA, United States",
        "description": "Requires 3+ years of experience.",
        "apply_link": "https://www.capitalonecareers.com/job/example",
    })
    assert not passes and reason == "location_outside_target_area"

    passes, reason, evidence = hard_filter.passes_final_hard_filters({
        "title": "Staff Product Analyst - Ads",
        "company": "Sirius XM",
        "location": "New York, NY",
        "description": "Requires 3+ years of experience.",
        "apply_link": "https://careers.siriusxm.com/jobs/123",
    })
    assert not passes and reason == "banned_seniority_title"

    passes, reason, evidence = hard_filter.passes_final_hard_filters({
        "title": "Product Analyst",
        "company": "Sirius XM",
        "location": "New York, NY",
        "description": "This role requires 7+ years of experience in product analytics.",
        "apply_link": "https://careers.siriusxm.com/jobs/123",
    })
    assert not passes and reason == "yoe_above_limit"

    passes, reason, evidence = hard_filter.passes_final_hard_filters({
        "title": "Product Analyst",
        "company": "ExampleCo",
        "location": "New York, NY",
        "description": "Candidates must be authorized to work without sponsorship.",
        "apply_link": "https://careers.exampleco.com/jobs/123",
    })
    assert not passes and reason == "no_sponsorship"

    passes, reason, evidence = hard_filter.passes_final_hard_filters({
        "title": "Data Analyst",
        "company": "ExampleCo",
        "location": "New York, NY",
        "description": "Base salary range is $90K-$120K.",
        "apply_link": "https://careers.exampleco.com/jobs/123",
    })
    assert not passes and reason == "salary_below_baseline"

    passes, reason, evidence = hard_filter.passes_final_hard_filters({
        "title": "Product Data Scientist",
        "company": "ExampleCo",
        "location": "New York, NY",
        "description": "Requires 3+ years of experience. Base salary range is $120K-$160K.",
        "apply_link": "https://careers.exampleco.com/jobs/123",
    })
    assert passes

    print("hard filter validation ok")


if __name__ == "__main__":
    main()
