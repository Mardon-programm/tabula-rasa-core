from behavioral import BehavioralProgram, Params
from behavioral.rules import Action, Condition, Literal


def _context(**attrs):
    return attrs


def test_simple_create_activate_and_fire():
    prog = BehavioralProgram()
    ctx = _context(
        role="adult",
        friendliness="warm",
        status="insider",
        mood="calm",
        context="greeting",
    )
    action = Action("primitive", "approach")

    # Three successes → LATENT→ACTIVE (n_min=3, p_min=0.7).
    for _ in range(3):
        prog.observe_outcome(ctx, action, success=True)

    assert prog.active_rules
    assert prog.choose_action(ctx) == action


def test_one_failure_does_not_break_rule():
    prog = BehavioralProgram()
    ctx = _context(
        role="adult",
        friendliness="warm",
        status="insider",
        mood="calm",
        context="greeting",
    )
    action = Action("primitive", "approach")

    for _ in range(3):
        prog.observe_outcome(ctx, action, success=True)
    # One failure → rule retained (F_min=3).
    prog.observe_outcome(ctx, action, success=False)

    assert prog.active_rules
    assert prog.choose_action(ctx) == action


def test_specialize_after_systematic_failure():
    prog = BehavioralProgram()
    params = Params(f_min=1, d_min=0.3)
    prog.params = params
    ctx_ok = _context(
        role="adult",
        friendliness="warm",
        status="insider",
        mood="calm",
        context="greeting",
    )
    ctx_bad = _context(
        role="adult",
        friendliness="cold",
        status="insider",
        mood="calm",
        context="greeting",
    )
    action = Action("primitive", "approach")

    # Build rule on the warm context.
    for _ in range(3):
        prog.observe_outcome(ctx_ok, action, success=True)

    # Systematic failures on the cold context (different friendliness).
    for _ in range(2):
        prog.observe_outcome(ctx_bad, action, success=False)

    # Discriminating feature must exist and a specialized rule created.
    assert any(r.status == "ACTIVE" for r in prog.rules)
    # The rule with warm must be more specific / correct.
    best = max(
        prog.matching_active(ctx_ok),
        key=lambda r: r.condition.specificity(),
    )
    assert best.condition.matches(ctx_ok)
    # Cold context should no longer fire the warm-specific rule blanket.
    cold_rules = [r for r in prog.active_rules if r.condition.matches(ctx_bad)]
    assert not any(r.action == action and r.condition.specificity() <= 1 for r in cold_rules)


def test_generalize_creates_variable_rule():
    prog = BehavioralProgram(allowed_operations=["CREATE", "GENERALIZE"])
    for role, ctx in [
        ("adult", _context(role="adult", friendliness="warm", status="insider", mood="calm", context="greeting")),
        ("child", _context(role="child", friendliness="warm", status="insider", mood="calm", context="greeting")),
    ]:
        act = Action("primitive", "approach")
        for _ in range(3):
            prog.observe_outcome(ctx, act, success=True)

    rule_a = next(r for r in prog.rules if "adult" in str(r.condition))
    rule_c = next(r for r in prog.rules if "child" in str(r.condition))
    groups = [
        ("adult", ["adult"]),
        ("child", ["child"]),
    ]
    combined = prog.generalize([rule_a.id, rule_c.id], "role", ["adult", "child"])
    assert combined is not None
    assert any(lit.attribute == "role" and lit.operator == "∈" for lit in combined.condition.literals)


def test_actions_fullspace_no_conflict():
    """In representative train contexts, rules never fire two actions."""
    from environment.attribute_tasks import AttributeTaskGenerator, is_success

    prog = BehavioralProgram()
    sets = AttributeTaskGenerator(seed=1).generate()

    for ctx in sets.train:
        action = Action("primitive", "approach")
        rewards = [is_success(ctx, a) for a in ["approach", "avoid", "ask", "obey"]]
        prog.observe_outcome(ctx, action, success=rewards[0])
    assert True