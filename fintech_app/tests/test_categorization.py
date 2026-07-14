from fintech_app import categorization


def test_groceries_match():
    assert categorization.categorize("Whole Foods Market", "grocery purchase", -50) == "groceries"


def test_dining_match_case_insensitive():
    assert categorization.categorize("STARBUCKS #123", "COFFEE", -5) == "dining"


def test_income_keyword_beats_positive_default():
    # explicit "payroll" keyword should be picked, not just the sign fallback
    assert categorization.categorize("Acme Corp Payroll", "direct deposit payroll", 2600) == "income"


def test_positive_amount_with_no_keyword_falls_back_to_income():
    assert categorization.categorize("Unknown Co", "misc credit", 42.00) == "income"


def test_negative_amount_with_no_keyword_falls_back_to_other():
    assert categorization.categorize("Mystery Vendor", "", -12.34) == "other"


def test_refund_description_is_categorized_by_keyword_not_sign():
    # A refund is a negative-amount-friendly word but this app has no
    # "refund" keyword rule, so it should NOT be miscategorized as income
    # just because the word sounds positive; it should fall back on sign.
    assert categorization.categorize("Random Store", "refund issued", -20) == "other"


def test_housing_beats_generic_shopping_when_both_could_apply():
    result = categorization.categorize("Skyline Property Management", "rent payment", -1450)
    assert result == "housing"


def test_priority_order_income_beats_transfer_when_blob_matches_both():
    # "payroll" (income) should win over "account transfer" (transfer)
    # when both keywords appear, since income is earlier in the priority list.
    blob_merchant = "Payroll Services account transfer"
    assert categorization.categorize(blob_merchant, "", 100) == "income"


def test_explain_reports_matched_keyword():
    result = categorization.explain("Starbucks", "coffee", -5)
    assert result["category"] == "dining"
    assert result["matched_keyword"] in ("starbucks", "coffee")
    assert result["used_fallback"] is False


def test_explain_reports_fallback_used():
    result = categorization.explain("Totally Unknown Merchant", "", 10)
    assert result["used_fallback"] is True
    assert result["category"] == "income"


def test_all_categories_includes_fallback_negative():
    assert "other" in categorization.ALL_CATEGORIES
    assert "income" in categorization.ALL_CATEGORIES


def test_every_keyword_list_is_nonempty():
    for category, keywords in categorization.CATEGORY_KEYWORDS.items():
        assert keywords, f"{category} has no keywords"


def test_category_priority_covers_all_keyword_categories():
    assert set(categorization.CATEGORY_PRIORITY) == set(categorization.CATEGORY_KEYWORDS.keys())
