"""Shared validation primitives — one source of truth for email + threshold."""

from app.validation import find_email, is_valid_email


class TestIsValidEmail:
    def test_accepts_plain_address(self):
        assert is_valid_email("wolf@gmail.com")

    def test_case_insensitive(self):
        assert is_valid_email("Max@Example.COM")

    def test_trims_surrounding_whitespace(self):
        assert is_valid_email("  a@b.co  ")

    def test_rejects_non_email(self):
        assert not is_valid_email("notanemail")

    def test_rejects_missing_tld(self):
        assert not is_valid_email("a@b")

    def test_rejects_one_char_tld(self):
        assert not is_valid_email("a@b.c")


class TestFindEmail:
    def test_finds_email_in_sentence(self):
        assert find_email("Ich habe notiert: wolf@gmail.com — korrekt?") == "wolf@gmail.com"

    def test_returns_none_without_email(self):
        assert find_email("hier ist keine Adresse") is None


class TestSingleSourceOfTruth:
    def test_threshold_is_shared_not_redefined(self):
        # manager and the extraction tool import the same constant, so tuning it in
        # one place can't silently fork the read-back policy.
        from app.conversation.manager import LOW_CONFIDENCE_THRESHOLD as from_manager
        from app.tools.extraction import LOW_CONFIDENCE_THRESHOLD as from_extraction
        from app.validation import LOW_CONFIDENCE_THRESHOLD as canonical

        assert from_manager is canonical
        assert from_extraction is canonical
