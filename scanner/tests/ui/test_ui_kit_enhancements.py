"""Unit tests for UI kit enhancement functions.

Tests the new shadow presets, glow effects, gradient helpers, and
font/radius constants added for the modern UI enhancements.
"""

import pytest

from scanner.tests.ui.conftest import make_app as _make_app
from scanner.ui.ui_kit import (  # noqa: F401 (import for type checking)
    ANIM_FAST,
    ANIM_NORMAL,
    COL_HIDE_ORDER,
    FS_XS,
    FS_BASE,
    FS_MD,
    FS_LG,
    FS_XL,
    FS_2XL,
    FS_HERO,
    RADIUS_SM,
    RADIUS_MD,
    RADIUS_LG,
    RADIUS_XL,
    RADIUS_XXL,
    RADIUS_XXXL,
    width_tier,
    _card_shadow,
    _deep_shadow,
    _soft_glow,
    _neon_glow,
    _inner_glow,
    _press_shadow,
    depth,
    glow,
    grad_v,
    label,
    score_color,
    rating_color,
    themed_dropdown,
    fuzzy_score,
    filter_actions,
    _glass_bg,
    _glass_card,
    _glass_border,
    OPTIONAL_COLS,
    ROW_H,
    hidden_cols,
    shimmer_cell,
    shimmer_row,
)


class TestRadiiAndFonts:
    """Test the new radius and font size constants."""

    def test_new_radius_constants_exist(self):
        """Verify the new RADIUS_XXL and RADIUS_XXXL constants are defined."""
        assert RADIUS_XXL == 20
        assert RADIUS_XXXL == 24

    def test_new_font_size_constants_exist(self):
        """Verify the new FS_XL and FS_HERO font sizes are defined."""
        assert FS_XL == 16  # card titles
        assert FS_HERO == 21  # hero headline


class TestWidthTier:
    """Test the width tier responsive function."""

    def test_width_tier_compact(self):
        """Width below 1400 should be compact."""
        assert width_tier(1280) == "compact"
        assert width_tier(1399) == "compact"

    def test_width_tier_standard(self):
        """Width 1400-1599 should be standard."""
        assert width_tier(1400) == "standard"
        assert width_tier(1599) == "standard"

    def test_width_tier_wide(self):
        """Width 1600 and above should be wide."""
        assert width_tier(1600) == "wide"
        assert width_tier(2000) == "wide"

    def test_width_tier_edge_cases(self):
        """Test edge cases for width tier function."""
        assert width_tier(None) == "wide"  # default to wide
        assert width_tier(0) == "wide"  # default to wide


class TestShadowPresets:
    """Test the new shadow preset functions."""

    def test_card_shadow_returns_list(self):
        """_card_shadow should return a list of BoxShadow."""
        shadows = _card_shadow()
        assert isinstance(shadows, list)
        assert len(shadows) == 2

    def test_card_shadow_has_correct_structure(self):
        """_card_shadow should have blur_radius, spread_radius, offset, color."""
        shadows = _card_shadow()
        for shadow in shadows:
            assert hasattr(shadow, 'blur_radius')
            assert hasattr(shadow, 'spread_radius')
            assert hasattr(shadow, 'offset')
            assert hasattr(shadow, 'color')

    def test_deep_shadow_returns_list(self):
        """_deep_shadow should return a list of BoxShadow."""
        shadows = _deep_shadow()
        assert isinstance(shadows, list)
        assert len(shadows) == 2

    def test_deep_shadow_deeper_than_card(self):
        """_deep_shadow should have larger blur than _card_shadow."""
        card = _card_shadow()
        deep = _deep_shadow()
        assert deep[0].blur_radius > card[0].blur_radius

    def test_soft_glow_returns_list(self):
        """_soft_glow should return a list of BoxShadow."""
        shadows = _soft_glow("#10b981")
        assert isinstance(shadows, list)
        assert len(shadows) == 1

    def test_soft_glow_has_correct_color(self):
        """_soft_glow should use the provided color."""
        shadows = _soft_glow("#ff0000")
        assert shadows[0].color == "#ff0000" or shadows[0].color == "#ffff0000"  # flet may parse differently

    def test_neon_glow_returns_list(self):
        """_neon_glow should return a list of BoxShadow."""
        shadows = _neon_glow("#10b981")
        assert isinstance(shadows, list)
        assert len(shadows) == 1

    def test_inner_glow_returns_list(self):
        """_inner_glow should return a list of BoxShadow with INNER blur style."""
        from flet import BoxShadow
        shadows = _inner_glow("#10b981")
        assert isinstance(shadows, list)
        assert len(shadows) == 1
        # Inner glow should have blur_style=BoxShadow.INNER
        assert shadows[0].blur_style.name == "INNER" or shadows[0].blur_style == 1

    def test_press_shadow_returns_list(self):
        """_press_shadow should return a list of BoxShadow."""
        shadows = _press_shadow()
        assert isinstance(shadows, list)
        assert len(shadows) == 1


class TestGlowEffects:
    """Test the glow effect functions."""

    def test_glow_returns_list(self):
        """glow should return a list of BoxShadow."""
        shadows = glow("#10b981", blur=26, spread=-4)
        assert isinstance(shadows, list)
        assert len(shadows) == 1

    def test_glow_has_correct_parameters(self):
        """glow should use the provided blur and spread."""
        shadows = glow("#10b981", blur=14, spread=0)
        assert shadows[0].blur_radius == 14
        assert shadows[0].spread_radius == 0

    def test_depth_returns_list(self):
        """depth should return a list of BoxShadow."""
        shadows = depth()
        assert isinstance(shadows, list)
        assert len(shadows) == 2

    def test_depth_has_inner_highlight(self):
        """depth should have an inner top highlight shadow."""
        shadows = depth()
        # The second shadow should be an inner shadow
        assert shadows[1].blur_style.name == "INNER" or shadows[1].blur_style == 1


class TestGradientHelpers:
    """Test the gradient helper functions."""

    def test_grad_v_returns_linear_gradient(self):
        """grad_v should return a LinearGradient."""
        from flet import LinearGradient
        gradient = grad_v(["#10b981", "#00ff00"])
        assert isinstance(gradient, LinearGradient)
        assert gradient.begin.x == 0.0 and gradient.begin.y == 0.0
        assert gradient.end.x == 0.0 and gradient.end.y == 1.0
        assert gradient.colors == ["#10b981", "#00ff00"]

    def test_grad_v_custom_begin_end(self):
        """grad_v should accept custom begin/end offsets."""
        from flet import LinearGradient
        gradient = grad_v(
            ["#10b981", "#00ff00"],
            begin=(1.0, 0.0),
            end=(0.0, 1.0),
        )
        assert isinstance(gradient, LinearGradient)
        assert gradient.begin.x == 1.0 and gradient.begin.y == 0.0
        assert gradient.end.x == 0.0 and gradient.end.y == 1.0


class TestScoreAndRatingColors:
    """Test the score and rating color functions."""

    def test_score_color_green(self):
        """Scores >= 70 should be green."""
        from scanner.shared.themes import THEMES
        c = THEMES["dark"]
        assert score_color(70, c) == c["green"]
        assert score_color(80, c) == c["green"]
        assert score_color(100, c) == c["green"]

    def test_score_color_lime(self):
        """Scores >= 50 and < 70 should be lime."""
        from scanner.shared.themes import THEMES
        c = THEMES["dark"]
        assert score_color(50, c) == c["lime"]
        assert score_color(60, c) == c["lime"]

    def test_score_color_orange(self):
        """Scores >= 30 and < 50 should be orange."""
        from scanner.shared.themes import THEMES
        c = THEMES["dark"]
        assert score_color(30, c) == c["orange"]
        assert score_color(40, c) == c["orange"]

    def test_score_color_red(self):
        """Scores < 30 should be red."""
        from scanner.shared.themes import THEMES
        c = THEMES["dark"]
        assert score_color(0, c) == c["red"]
        assert score_color(20, c) == c["red"]

    def test_rating_color_excellent(self):
        """EXCELLENT rating should be green."""
        from scanner.shared.themes import THEMES
        c = THEMES["dark"]
        assert rating_color("EXCELLENT", c) == c["green"]

    def test_rating_color_good(self):
        """GOOD rating should be lime."""
        from scanner.shared.themes import THEMES
        c = THEMES["dark"]
        assert rating_color("GOOD", c) == c["lime"]

    def test_rating_color_moderate(self):
        """MODERATE rating should be orange."""
        from scanner.shared.themes import THEMES
        c = THEMES["dark"]
        assert rating_color("MODERATE", c) == c["orange"]

    def test_rating_color_poor(self):
        """POOR rating should be red."""
        from scanner.shared.themes import THEMES
        c = THEMES["dark"]
        assert rating_color("POOR", c) == c["red"]


class TestFuzzyScoreAndFilter:
    """Test the fuzzy matching functions (already existed, ensure no regression)."""

    def test_fuzzy_score_exact_match(self):
        """Exact match should score 1000."""
        assert fuzzy_score("dashboard", "dashboard") == 1000.0

    def test_fuzzy_score_prefix_match(self):
        """Prefix match should work."""
        result = fuzzy_score("dash", "dashboard")
        assert result is not None
        assert result > 0

    def test_fuzzy_score_no_match(self):
        """Missing letter should return None."""
        assert fuzzy_score("zxq", "dashboard") is None

    def test_filter_actions_empty_query(self):
        """Empty query should return all actions."""
        from scanner.ui.ui_kit import PaletteAction
        actions = [
            PaletteAction("a", "A", ""),
            PaletteAction("b", "B", ""),
        ]
        result = filter_actions("", actions)
        assert len(result) == 2

    def test_filter_actions_with_query(self):
        """Query should filter actions."""
        from scanner.ui.ui_kit import PaletteAction
        actions = [
            PaletteAction("go-dashboard", "Go to Dashboard", ""),
            PaletteAction("run-stop", "Run scan", ""),
        ]
        result = filter_actions("go", actions)
        assert len(result) == 1
        assert result[0].action_id == "go-dashboard"


class TestLabelFunction:
    """Test the label function."""

    def test_label_uppercase(self):
        """Label should return uppercase text."""
        result = label("test", "#ffffff")
        assert result.value == "TEST"

    def test_label_small_size(self):
        """Label with FS_XS size should work."""
        result = label("test", "#ffffff", size=FS_XS)
        assert result.value == "TEST"

    def test_label_large_size(self):
        """Label with FS_HERO size should work."""
        result = label("test", "#ffffff", size=FS_HERO)
        assert result.value == "TEST"


class TestThemedDropdown:
    """Test the themed_dropdown function."""

    def test_dropdown_has_correct_defaults(self):
        """Themed dropdown should have default width/height."""
        from scanner.shared.themes import THEMES
        c = THEMES["dark"]
        dropdown = themed_dropdown(options=[1, 2, 3], value=1, c=c)
        assert dropdown.width == 160
        assert dropdown.height == 42

    def test_dropdown_colors(self):
        """Themed dropdown should use theme colors."""
        from scanner.shared.themes import THEMES
        c = THEMES["dark"]
        dropdown = themed_dropdown(options=[1, 2, 3], value=1, c=c)
        assert dropdown.bgcolor == c["option_bg"]
        assert dropdown.color == c["text"]
        assert dropdown.border_color == c["border"]


class TestGlassEffects:
    """Test the glass effect helper functions."""

    def test_glass_bg_return_string(self):
        """_glass_bg should return a color string."""
        result = _glass_bg()
        assert isinstance(result, str)
        # Should be translucent white
        assert "0.065" in result or "65" in result

    def test_glass_card_return_gradient(self):
        """_glass_card should return a LinearGradient."""
        from flet import LinearGradient
        result = _glass_card()
        assert isinstance(result, LinearGradient)

    def test_glass_border_return_border(self):
        """_glass_border should return a Border."""
        from flet import Border
        result = _glass_border()
        assert isinstance(result, Border)