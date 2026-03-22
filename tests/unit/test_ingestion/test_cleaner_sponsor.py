"""Unit tests for reviewmind.ingestion.cleaner and reviewmind.ingestion.sponsor."""

from __future__ import annotations

import pytest

from reviewmind.ingestion.cleaner import (
    _BRACKET_ARTEFACT_RE,
    _EMAIL_RE,
    _HORIZONTAL_WS_RE,
    _HTML_COMMENT_RE,
    _HTML_TAG_RE,
    _INVISIBLE_CHARS_RE,
    _MD_IMAGE_RE,
    _MD_LINK_RE,
    _MULTI_NEWLINE_RE,
    _NAV_FRAGMENTS_RE,
    _REPEATED_PUNCT_RE,
    _TIMESTAMP_RE,
    _URL_RE,
    MIN_CLEAN_LENGTH,
    clean_text,
)
from reviewmind.ingestion.sponsor import (
    _EN_PATTERNS,
    _RU_PATTERNS,
    ALL_PATTERNS,
    SponsorDetectionResult,
    _find_matches,
    detect_sponsor,
    detect_sponsor_detailed,
)

# ═══════════════════════════════════════════════════════════════════════════
# CLEANER TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestCleanerConstants:
    """Test exported constants."""

    def test_min_clean_length_positive(self) -> None:
        assert MIN_CLEAN_LENGTH > 0

    def test_min_clean_length_value(self) -> None:
        assert MIN_CLEAN_LENGTH == 50


class TestHtmlRemoval:
    """Test HTML tag and comment removal."""

    def test_simple_tags(self) -> None:
        assert _HTML_TAG_RE.sub("", "<p>Hello</p>") == "Hello"

    def test_self_closing_tag(self) -> None:
        assert _HTML_TAG_RE.sub("", "Hello<br/>World") == "HelloWorld"

    def test_nested_tags(self) -> None:
        assert _HTML_TAG_RE.sub("", "<div><b>Bold</b></div>") == "Bold"

    def test_html_comment(self) -> None:
        assert _HTML_COMMENT_RE.sub("", "before<!-- comment -->after") == "beforeafter"

    def test_multiline_comment(self) -> None:
        text = "a<!--\nmultiline\ncomment\n-->b"
        assert _HTML_COMMENT_RE.sub("", text) == "ab"


class TestTimestampRemoval:
    """Test timestamp regex matching."""

    @pytest.mark.parametrize(
        "ts",
        [
            "00:01:23",
            "1:23:45",
            "12:34",
            "0:12",
            "[00:01]",
            "(12:34)",
            "12:34:56.789",
            "00:00:01,500",
        ],
    )
    def test_matches(self, ts: str) -> None:
        assert _TIMESTAMP_RE.search(ts), f"Pattern should match '{ts}'"


class TestBracketArtefacts:
    """Test [Music], [Applause] etc. removal."""

    @pytest.mark.parametrize(
        "artefact",
        ["[Music]", "[Applause]", "[Laughter]", "[Silence]", "[Inaudible]", "[Музыка]", "[Аплодисменты]", "[Смех]"],
    )
    def test_matches(self, artefact: str) -> None:
        assert _BRACKET_ARTEFACT_RE.search(artefact)

    def test_case_insensitive(self) -> None:
        assert _BRACKET_ARTEFACT_RE.search("[MUSIC]")
        assert _BRACKET_ARTEFACT_RE.search("[music]")


class TestUrlRemoval:
    """Test URL pattern."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com",
            "http://www.example.com/path?q=1",
            "https://youtu.be/abc123",
            "ftp://files.example.org/data",
        ],
    )
    def test_matches(self, url: str) -> None:
        assert _URL_RE.search(url)


class TestEmailRemoval:
    """Test email pattern."""

    def test_simple_email(self) -> None:
        assert _EMAIL_RE.search("user@example.com")

    def test_complex_email(self) -> None:
        assert _EMAIL_RE.search("first.last+tag@sub.example.co.uk")


class TestInvisibleChars:
    """Test zero-width / invisible character removal."""

    def test_zero_width_space(self) -> None:
        assert _INVISIBLE_CHARS_RE.search("\u200b")

    def test_bom(self) -> None:
        assert _INVISIBLE_CHARS_RE.search("\ufeff")

    def test_soft_hyphen(self) -> None:
        assert _INVISIBLE_CHARS_RE.search("\u00ad")


class TestWhitespaceCollapse:
    """Test whitespace normalisation."""

    def test_spaces_and_tabs(self) -> None:
        assert _HORIZONTAL_WS_RE.sub(" ", "Hello  \t world") == "Hello world"

    def test_preserves_newlines(self) -> None:
        result = _HORIZONTAL_WS_RE.sub(" ", "line1\n  line2")
        assert "\n" in result

    def test_multi_newline_collapse(self) -> None:
        assert _MULTI_NEWLINE_RE.sub("\n\n", "a\n\n\n\nb") == "a\n\nb"


class TestMarkdown:
    """Test markdown pattern handling."""

    def test_md_image_removal(self) -> None:
        assert _MD_IMAGE_RE.sub("", "![alt text](http://img.png)") == ""

    def test_md_link_keeps_text(self) -> None:
        assert _MD_LINK_RE.sub(r"\1", "[click here](http://example.com)") == "click here"


class TestNavFragments:
    """Test navigation fragment removal."""

    @pytest.mark.parametrize(
        "fragment",
        ["\nShare\n", "\nSubscribe\n", "\nПодписаться\n", "\nAdvertisement\n", "\nРеклама\n"],
    )
    def test_fragment_matches(self, fragment: str) -> None:
        assert _NAV_FRAGMENTS_RE.search(fragment)


class TestRepeatedPunctuation:
    """Test repeated punctuation collapsing."""

    def test_exclamation(self) -> None:
        assert _REPEATED_PUNCT_RE.sub(r"\1", "wow!!!") == "wow!"

    def test_question(self) -> None:
        assert _REPEATED_PUNCT_RE.sub(r"\1", "really????") == "really?"

    def test_dots(self) -> None:
        assert _REPEATED_PUNCT_RE.sub(r"\1", "hmm.....") == "hmm."


class TestCleanText:
    """Test the main clean_text function end-to-end."""

    def _long(self, text: str) -> str:
        """Pad text to exceed MIN_CLEAN_LENGTH."""
        padding = " This is padding text to make the string long enough for the cleaner to accept it."
        return text + padding * 2

    def test_empty_string(self) -> None:
        assert clean_text("") == ""

    def test_none_like(self) -> None:
        assert clean_text("   ") == ""

    def test_too_short_after_cleaning(self) -> None:
        assert clean_text("Hi") == ""

    def test_html_tags_removed(self) -> None:
        raw = self._long("<p>Hello  \t world</p>")
        result = clean_text(raw)
        assert "<p>" not in result
        assert "</p>" not in result
        assert "Hello" in result
        assert "world" in result

    def test_task_step_1(self) -> None:
        """From task test step: clean_text('<p>Hello  \\t world</p>') → 'Hello world'."""
        raw = self._long("<p>Hello  \t world</p>")
        result = clean_text(raw)
        assert result.startswith("Hello world")

    def test_timestamps_removed(self) -> None:
        raw = self._long("00:01:23 This is what the speaker said at this point.")
        result = clean_text(raw)
        assert "00:01:23" not in result
        assert "speaker said" in result

    def test_html_entities_decoded(self) -> None:
        raw = self._long("Tom &amp; Jerry are &quot;friends&quot;")
        result = clean_text(raw)
        assert "Tom & Jerry" in result
        assert '"friends"' in result

    def test_unicode_normalised_nfkc(self) -> None:
        # ﬁ (U+FB01) → fi in NFKC
        raw = self._long("The ﬁlm was great and had fantastic visuals overall.")
        result = clean_text(raw)
        assert "fi" in result  # normalised

    def test_bracket_artefacts_removed(self) -> None:
        raw = self._long("[Music] The product works great [Applause] when used properly honestly.")
        result = clean_text(raw)
        assert "[Music]" not in result
        assert "[Applause]" not in result
        assert "product works great" in result

    def test_urls_removed_by_default(self) -> None:
        raw = self._long("Visit https://example.com for more info on this product!")
        result = clean_text(raw)
        assert "https://example.com" not in result
        assert "Visit" in result

    def test_urls_kept_when_disabled(self) -> None:
        raw = self._long("Visit https://example.com for more info on product really.")
        result = clean_text(raw, remove_urls=False)
        assert "https://example.com" in result

    def test_emails_removed_by_default(self) -> None:
        raw = self._long("Contact us at info@example.com for product support really nice.")
        result = clean_text(raw)
        assert "info@example.com" not in result

    def test_emails_kept_when_disabled(self) -> None:
        raw = self._long("Contact us at info@example.com for product support really nice.")
        result = clean_text(raw, remove_emails=False)
        assert "info@example.com" in result

    def test_invisible_chars_removed(self) -> None:
        raw = self._long("Hello\u200bWorld\ufeffToday is a great day for products!")
        result = clean_text(raw)
        assert "\u200b" not in result
        assert "\ufeff" not in result

    def test_whitespace_collapsed(self) -> None:
        raw = self._long("Hello     world    this   is   a   test   of   whitespace")
        result = clean_text(raw)
        assert "  " not in result

    def test_multi_newlines_collapsed(self) -> None:
        raw = self._long("paragraph one\n\n\n\n\nparagraph two is here")
        result = clean_text(raw)
        assert "\n\n\n" not in result
        assert "paragraph one" in result

    def test_md_images_removed(self) -> None:
        raw = self._long("Text before ![logo](http://img.png) and text after is good.")
        result = clean_text(raw)
        assert "![logo]" not in result
        assert "Text before" in result

    def test_md_links_converted(self) -> None:
        raw = self._long("Check [this review](http://example.com) for a deeper analysis of the product.")
        result = clean_text(raw)
        assert "this review" in result
        assert "http://example.com" not in result  # URL removed

    def test_repeated_punctuation(self) -> None:
        raw = self._long("Wow!!! Amazing product??? Really great... it works.")
        result = clean_text(raw)
        assert "!!!" not in result
        assert "???" not in result
        assert "..." not in result

    def test_preserves_normal_text(self) -> None:
        raw = (
            "This is a perfectly normal review about Sony WH-1000XM5 headphones. "
            "They are great for noise cancelling and music listening."
        )
        result = clean_text(raw)
        assert "Sony WH-1000XM5" in result
        assert "noise cancelling" in result

    def test_russian_text(self) -> None:
        raw = "Отличные наушники с хорошим шумоподавлением. Качество звука превосходное, батарея держит долго."
        result = clean_text(raw)
        assert "наушники" in result
        assert "шумоподавлением" in result

    def test_combined_cleanup(self) -> None:
        """Complex text with multiple issues."""
        raw = (
            "<div>[Music] 00:01:23 Hello  \t world</div>\n"
            "Visit https://example.com\n"
            "<!-- hidden comment -->\n"
            "\n\n\n\n"
            "This is the actual content of the review. It discusses the pros and cons of the product in detail."
        )
        result = clean_text(raw)
        assert "<div>" not in result
        assert "[Music]" not in result
        assert "00:01:23" not in result
        assert "https://example.com" not in result
        assert "hidden comment" not in result
        assert "\n\n\n" not in result
        assert "actual content" in result

    def test_nav_fragments_removed(self) -> None:
        raw = self._long("\nShare\nThe product review continues here with good details.")
        result = clean_text(raw)
        # Share should be removed as nav fragment
        assert "product review continues" in result


# ═══════════════════════════════════════════════════════════════════════════
# SPONSOR DETECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestSponsorPatterns:
    """Test that pattern lists are populated."""

    def test_en_patterns_not_empty(self) -> None:
        assert len(_EN_PATTERNS) > 0

    def test_ru_patterns_not_empty(self) -> None:
        assert len(_RU_PATTERNS) > 0

    def test_all_patterns_combined(self) -> None:
        assert len(ALL_PATTERNS) == len(_EN_PATTERNS) + len(_RU_PATTERNS)


class TestDetectSponsorEnglish:
    """Test sponsor detection — English patterns."""

    def test_sponsored_by(self) -> None:
        assert detect_sponsor("This video is sponsored by NordVPN") is True

    def test_brought_to_you(self) -> None:
        assert detect_sponsor("This content is brought to you by Squarespace") is True

    def test_promo_code(self) -> None:
        assert detect_sponsor("Use promo code TECH10 for 15% off") is True

    def test_coupon_code(self) -> None:
        assert detect_sponsor("Get a coupon code for 20% discount") is True

    def test_affiliate_link(self) -> None:
        assert detect_sponsor("Affiliate link in the description below") is True

    def test_check_link_description(self) -> None:
        assert detect_sponsor("Check out the link in the description") is True

    def test_hashtag_ad(self) -> None:
        assert detect_sponsor("Great product! #ad") is True

    def test_hashtag_sponsored(self) -> None:
        assert detect_sponsor("Review #sponsored") is True

    def test_paid_partnership(self) -> None:
        assert detect_sponsor("This is a paid partnership with BrandX") is True

    def test_special_offer(self) -> None:
        assert detect_sponsor("Special offer just for our viewers") is True

    def test_thanks_for_sponsoring(self) -> None:
        assert detect_sponsor("Thanks to BrandX for sponsoring this video") is True

    def test_in_partnership_with(self) -> None:
        assert detect_sponsor("In partnership with Samsung") is True

    def test_use_my_link(self) -> None:
        assert detect_sponsor("Use my link to sign up") is True

    def test_exclusive_deal(self) -> None:
        assert detect_sponsor("Exclusive deal for subscribers") is True

    def test_honest_review_not_sponsored(self) -> None:
        assert detect_sponsor("Honest review of headphones") is False

    def test_normal_review_text(self) -> None:
        text = "The Sony WH-1000XM5 has excellent noise cancelling and comfortable fit."
        assert detect_sponsor(text) is False

    def test_technical_discussion(self) -> None:
        text = "Battery life lasts about 30 hours, Bluetooth 5.2 with LDAC codec."
        assert detect_sponsor(text) is False


class TestDetectSponsorRussian:
    """Test sponsor detection — Russian patterns."""

    def test_promo_code(self) -> None:
        """Task test step: detect_sponsor('Промокод TECH10 по ссылке в описании') → True"""
        assert detect_sponsor("Промокод TECH10 по ссылке в описании") is True

    def test_sponsor_word(self) -> None:
        assert detect_sponsor("Спонсором выпуска выступает компания Brand") is True

    def test_reklama(self) -> None:
        assert detect_sponsor("Реклама. Информация по ссылке.") is True

    def test_reklamaya_integraciya(self) -> None:
        assert detect_sponsor("Рекламная интеграция с NordVPN") is True

    def test_blagodarim_sponsora(self) -> None:
        assert detect_sponsor("Благодарим спонсора этого видео") is True

    def test_skidka_po_ssylke(self) -> None:
        assert detect_sponsor("Скидка по ссылке в описании!") is True

    def test_ssylka_v_opisanii(self) -> None:
        assert detect_sponsor("Ссылка в описании к видео") is True

    def test_pri_podderzhke(self) -> None:
        assert detect_sponsor("При поддержке компании Samsung") is True

    def test_partnerskiy(self) -> None:
        assert detect_sponsor("Партнёрский материал с Яндексом") is True

    def test_kupon(self) -> None:
        assert detect_sponsor("Купон на скидку 15%") is True

    def test_zakaznoy_obzor(self) -> None:
        assert detect_sponsor("Заказной обзор нового смартфона") is True

    def test_normal_russian_review(self) -> None:
        text = "Отличные наушники с хорошим шумоподавлением"
        assert detect_sponsor(text) is False

    def test_normal_russian_technical(self) -> None:
        text = "Автономность составляет 30 часов, Bluetooth 5.2 с поддержкой LDAC"
        assert detect_sponsor(text) is False


class TestDetectSponsorEdgeCases:
    """Test edge cases."""

    def test_empty_string(self) -> None:
        assert detect_sponsor("") is False

    def test_whitespace_only(self) -> None:
        assert detect_sponsor("   ") is False

    def test_threshold_parameter(self) -> None:
        # Text with exactly one match
        text = "Use promo code TECH for savings"
        assert detect_sponsor(text, threshold=1) is True
        assert detect_sponsor(text, threshold=2) is False

    def test_multiple_matches_threshold(self) -> None:
        text = "Sponsored by NordVPN. Use promo code TECH for a special offer!"
        assert detect_sponsor(text, threshold=1) is True
        assert detect_sponsor(text, threshold=2) is True
        assert detect_sponsor(text, threshold=3) is True

    def test_case_insensitive(self) -> None:
        assert detect_sponsor("SPONSORED BY BRANDX") is True
        assert detect_sponsor("Sponsored By BrandX") is True
        assert detect_sponsor("sponsored by brandx") is True

    def test_mixed_languages(self) -> None:
        text = "This video is sponsored by NordVPN. Промокод TECH10."
        assert detect_sponsor(text) is True


class TestDetectSponsorDetailed:
    """Test detect_sponsor_detailed."""

    def test_sponsored_result(self) -> None:
        result = detect_sponsor_detailed("This video is sponsored by NordVPN")
        assert result.is_sponsored is True
        assert len(result.matched_patterns) >= 1
        assert result.confidence > 0.0

    def test_not_sponsored_result(self) -> None:
        result = detect_sponsor_detailed("Honest review of headphones")
        assert result.is_sponsored is False
        assert result.matched_patterns == []
        assert result.confidence == 0.0

    def test_empty_string(self) -> None:
        result = detect_sponsor_detailed("")
        assert result.is_sponsored is False
        assert result.matched_patterns == []
        assert result.confidence == 0.0

    def test_confidence_scaling(self) -> None:
        # Multiple matches → higher confidence
        text = "Sponsored by NordVPN. Use promo code TECH for a special offer! Check out the link in the description!"
        result = detect_sponsor_detailed(text)
        assert result.confidence > 0.5

    def test_confidence_max_1(self) -> None:
        text = (
            "Sponsored by NordVPN. Use promo code TECH for a special offer! "
            "Check out the link in the description. In partnership with BrandX. "
            "Thanks to BrandY for sponsoring this."
        )
        result = detect_sponsor_detailed(text)
        assert result.confidence <= 1.0

    def test_threshold_respected(self) -> None:
        text = "This video is sponsored by NordVPN"
        r1 = detect_sponsor_detailed(text, threshold=1)
        r2 = detect_sponsor_detailed(text, threshold=5)
        assert r1.is_sponsored is True
        assert r2.is_sponsored is False
        # Both should still have the same matched patterns
        assert r1.matched_patterns == r2.matched_patterns


class TestSponsorDetectionResult:
    """Test the SponsorDetectionResult dataclass."""

    def test_frozen(self) -> None:
        result = SponsorDetectionResult(is_sponsored=True, matched_patterns=["test"], confidence=0.5)
        with pytest.raises(AttributeError):
            result.is_sponsored = False  # type: ignore[misc]

    def test_default_values(self) -> None:
        result = SponsorDetectionResult(is_sponsored=False)
        assert result.matched_patterns == []
        assert result.confidence == 0.0


class TestFindMatches:
    """Test the internal _find_matches helper."""

    def test_no_matches(self) -> None:
        assert _find_matches("Normal product review text") == []

    def test_deduplication(self) -> None:
        # Same pattern matching twice should only appear once
        text = "Sponsored by X. sponsored by Y."
        matches = _find_matches(text)
        lower_matches = [m.lower() for m in matches]
        assert len(lower_matches) == len(set(lower_matches))


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION / EXPORTS
# ═══════════════════════════════════════════════════════════════════════════


class TestIngestionExports:
    """Test that ingestion __init__.py exports expected symbols."""

    def test_clean_text_importable(self) -> None:
        from reviewmind.ingestion import clean_text as ct

        assert callable(ct)

    def test_detect_sponsor_importable(self) -> None:
        from reviewmind.ingestion import detect_sponsor as ds

        assert callable(ds)

    def test_detect_sponsor_detailed_importable(self) -> None:
        from reviewmind.ingestion import detect_sponsor_detailed as dsd

        assert callable(dsd)

    def test_min_clean_length_importable(self) -> None:
        from reviewmind.ingestion import MIN_CLEAN_LENGTH as mcl

        assert mcl == 50

    def test_all_patterns_importable(self) -> None:
        from reviewmind.ingestion import ALL_PATTERNS as ap

        assert len(ap) > 0

    def test_result_importable(self) -> None:
        from reviewmind.ingestion import SponsorDetectionResult as sdr

        assert sdr is not None


class TestIntegrationScenarios:
    """End-to-end scenarios combining cleaner + sponsor detection."""

    def test_clean_then_detect_sponsored(self) -> None:
        raw = "<p>This video is sponsored by NordVPN. Use promo code TECH10!</p>"
        cleaned = clean_text(raw)
        assert cleaned  # long enough after cleaning? Pad if needed
        if cleaned:
            assert detect_sponsor(cleaned) is True

    def test_clean_then_detect_not_sponsored(self) -> None:
        raw = (
            "The Sony WH-1000XM5 is one of the best noise-cancelling headphones on the market. "
            "It offers 30 hours of battery life, excellent sound quality where you can hear "
            "every nuance in your music."
        )
        cleaned = clean_text(raw)
        assert cleaned != ""
        assert detect_sponsor(cleaned) is False

    def test_sponsored_youtube_transcript(self) -> None:
        """Simulate YouTube transcript with sponsor segment."""
        raw = (
            "[Music] 00:01:23 Hey guys welcome to today's review! "
            "Before we start, this video is sponsored by NordVPN. "
            "Use promo code TECH10 for 15% off! Check out the link in the description. "
            "Now let's talk about the Sony WH-1000XM5 headphones. "
            "They have amazing noise cancelling capabilities."
        )
        cleaned = clean_text(raw)
        assert "[Music]" not in cleaned
        assert "00:01:23" not in cleaned
        assert detect_sponsor(cleaned) is True

    def test_non_sponsored_reddit_post(self) -> None:
        """Simulate Reddit post without sponsorship."""
        raw = (
            "I've been using the Sony WH-1000XM5 for about 3 months now. "
            "Here are my thoughts:\n\n"
            "Pros: Amazing noise cancelling, comfortable for long sessions, "
            "great sound quality.\n\n"
            "Cons: Touch controls can be finicky, case is quite bulky. "
            "Overall I would recommend them."
        )
        cleaned = clean_text(raw)
        assert cleaned != ""
        assert detect_sponsor(cleaned) is False

    def test_russian_sponsored_video(self) -> None:
        """Russian YouTube review with sponsor."""
        raw = (
            "[Музыка] Привет, друзья! Сегодня обзор наушников Sony WH-1000XM5. "
            "Но сначала — спонсор этого видео. Благодарим спонсора — компанию NordVPN. "
            "Промокод TECH10 по ссылке в описании. Скидка по коду 15 процентов! "
            "Теперь перейдём к обзору. Шумоподавление на уровне, "
            "звук отличный, батарея держит долго."
        )
        cleaned = clean_text(raw)
        assert "[Музыка]" not in cleaned
        assert detect_sponsor(cleaned) is True
        detail = detect_sponsor_detailed(cleaned)
        assert detail.confidence > 0.5

    def test_false_positive_rate(self) -> None:
        """Ensure common product review phrases are not flagged."""
        samples = [
            "The battery life is excellent, lasting 30 hours on a single charge.",
            "Build quality is premium with aluminum construction.",
            "Шумоподавление работает отлично даже в метро.",
            "Compared to the previous model, this is a significant upgrade.",
            "I would recommend this to anyone looking for quality headphones.",
            "Цена кусается, но качество того стоит.",
            "The microphone quality during calls was surprisingly good.",
            "Посмотрите мой подробный обзор этих наушников в другом видео.",
            "Sound signature is warm with slightly boosted bass.",
            "Удобные для длительного ношения, не давят на уши.",
        ]
        false_positives = sum(1 for s in samples if detect_sponsor(s))
        # Acceptance criteria: false positive rate < 10%
        assert false_positives / len(samples) < 0.10, f"False positive rate too high: {false_positives}/{len(samples)}"
