from environment.attribute_tasks import (
    ACTIONS,
    AttributeTaskGenerator,
    all_contexts,
    classify,
    is_t1,
    is_t2,
    is_t3,
    is_train_eligible,
    oracle_action,
    validate_oracle,
)


def test_oracle_total_and_consistent():
    validate_oracle()
    contexts = all_contexts()
    assert len(contexts) == 72
    for ctx in contexts:
        assert oracle_action(ctx) in ACTIONS


def test_train_eligibility():
    train_contexts = [ctx for ctx in all_contexts() if is_train_eligible(ctx)]
    assert all(ctx["role"] != "elder" for ctx in train_contexts)
    assert all(
        not (ctx["mood"] == "upset" and ctx["context"] == "conflict")
        for ctx in train_contexts
    )
    assert len(train_contexts) == 40


def test_axis_partition_disjoint():
    contexts = all_contexts()
    t1 = [c for c in contexts if is_t1(c)]
    t2 = [c for c in contexts if is_t2(c)]
    t3 = [c for c in contexts if is_t3(c)]

    assert t1 and t2 and t3
    assert not set(map(str, t1)) & set(map(str, t2))
    assert not set(map(str, t1)) & set(map(str, t3))
    assert not set(map(str, t2)) & set(map(str, t3))

    for ctx in contexts:
        cls = classify(ctx)
        if cls == "t1":
            assert is_t1(ctx)
        elif cls == "t2":
            assert is_t2(ctx)
        elif cls == "t3":
            assert is_t3(ctx)


def test_generator_axes_do_not_leak_train():
    gen = AttributeTaskGenerator(seed=7)
    sets = gen.generate()

    assert len(sets.train) == 600
    assert len(sets.known) == 120
    assert len(sets.t1) == 60
    assert len(sets.t2) == 60
    assert len(sets.t3) == 40

    for ctx in sets.t1 + sets.t2 + sets.t3:
        assert not is_train_eligible(ctx), f"test task leaked into train space: {ctx}"