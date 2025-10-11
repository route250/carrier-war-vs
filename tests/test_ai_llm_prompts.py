from server.schemas import MatchStatePayload, PayloadUnit, Position, SideViewPayload, TurnLog
from server.services.ai_llm_prompts import match_state_payload_to_text


def test_match_state_payload_to_text_produces_readable_summary():
    payload = MatchStatePayload(
        match_id="match-1",
        status="active",
        turn=3,
        max_turn=30,
        waiting_for="you",
        map_w=30,
        map_h=30,
        units=SideViewPayload(
            carrier=PayloadUnit(
                id="C1",
                hp=100,
                max_hp=100,
                x=5,
                y=7,
                fuel=999,
                speed=2,
                vision=4,
                target=Position(x=7, y=11),
            ),
            squadrons=[
                PayloadUnit(
                    id="SQ-1",
                    hp=25,
                    max_hp=40,
                    x=8,
                    y=9,
                    state="onboard",
                    fuel=12,
                    speed=4,
                    vision=5,
                ),
                PayloadUnit(
                    id="SQ-2",
                    hp=25,
                    max_hp=40,
                    x=8,
                    y=9,
                    state="outbound",
                    fuel=12,
                    speed=4,
                    vision=5,
                    target=Position(x=12, y=10),
                ),
                PayloadUnit(
                    id="SQ-3",
                    hp=25,
                    max_hp=40,
                    x=8,
                    y=9,
                    state="engaging",
                    fuel=12,
                    speed=4,
                    vision=5,
                    target=Position(x=12, y=10),
                ),
                PayloadUnit(
                    id="SQ-4",
                    hp=20,
                    max_hp=40,
                    x=8,
                    y=9,
                    state="returning",
                    fuel=12,
                    speed=4,
                    vision=5,
                    target=Position(x=12, y=10),
                ),
                PayloadUnit(
                    id="SQ-5",
                    hp=0,
                    max_hp=40,
                    x=8,
                    y=9,
                    state="lost",
                    fuel=12,
                    speed=4,
                    vision=5,
                ),
            ],
            turn_visible=["EN-SQ-1"],
        ),
        intel=SideViewPayload(
            carrier=PayloadUnit(
                id="EN-C1",
                hp=80,
                max_hp=100,
            ),
            squadrons=[
                PayloadUnit(
                    id="EN-SQ-1",
                    hp=30,
                    max_hp=40,
                    x=15,
                    y=16,
                    state="engaging",
                    fuel=12,
                    speed=4,
                    vision=5,
                    x0=5,
                    y0=3,
                )
            ],
        ),
        logs=[
            TurnLog(
                step=1,
                unit_id="SQ-1",
                unit_pos=Position(x=8, y=9),
                report="found",
                target_id="EN-C1",
                target_pos=Position(x=15, y=16),
                target_from=None,
                value=None,
            )
        ],
        result=None,
    )

    text = match_state_payload_to_text(payload)

    print("---")
    print(text)
    print("---")

    # Basic structural expectations based on current formatter
    assert "Below is the current battle situation." in text
    assert "Follow all constraints and respond using the required JSON output format." in text
    assert "Turn 3 of 30." in text
    assert "Your forces:" in text

    # Carrier and squadron details should be present
    assert "- Your carrier C1" in text
    assert "position (5, 7)" in text
    assert "HP 100/100" in text
    # SQ-1 is on board in the provided payload
    assert "on board and waiting on the carrier" in text

    assert "Enemy intelligence:" in text
    assert "- Enemy carrier EN-C1" in text

    # Recent events include the spotted report with positions
    assert "Recent events:" in text
    assert "SQ-1 spotted EN-C1 at (15, 16) while at (8, 9)" in text

    assert "A structured JSON snapshot is appended after this summary." in text


def _make_payload_for_reports(report_items):
    return MatchStatePayload(
        match_id="m-2",
        status="active",
        turn=1,
        max_turn=10,
        waiting_for="none",
        map_w=20,
        map_h=20,
        units=SideViewPayload(
            carrier=PayloadUnit(id="C2", hp=50, max_hp=100, x=2, y=2),
            squadrons=[
                PayloadUnit(id="S1", hp=10, max_hp=40, x=3, y=3, state="outbound", fuel=5, speed=4, vision=5, target=Position(x=5,y=5)),
                PayloadUnit(id="S2", hp=0, max_hp=40, x=0, y=0, state="lost", fuel=0, speed=4, vision=5),
                PayloadUnit(id="S3", hp=30, max_hp=40, x=4, y=4, state="returning", fuel=10, speed=4, vision=5),
            ],
        ),
        intel=SideViewPayload(
            carrier=PayloadUnit(id="EN-C2", hp=0, max_hp=100, x=None, y=None),
            squadrons=[
                PayloadUnit(id="EN-S1", hp=20, max_hp=40, x=7, y=7, x0=6, y0=6),
            ],
        ),
        logs=report_items,
        result=None,
    )


def test_all_turnlog_report_types_are_rendered():
    # Uses top-level imports
    # Create a variety of logs for each report type
    logs = [
        TurnLog(1, "S1", Position(x=3,y=3), "target", target_id="EN-C2", target_pos=Position(x=7,y=7)),
        TurnLog(2, "S1", Position(x=3,y=3), "returning"),
        TurnLog(3, "S1", Position(x=3,y=3), "attack", target_id="EN-S1"),
        TurnLog(4, "S1", Position(x=3,y=3), "hit", value=5, target_id="EN-S1"),
        TurnLog(5, "S2", None, "lost"),
        TurnLog(6, "S3", Position(x=4,y=4), "landed"),
        TurnLog(7, "S1", Position(x=3,y=3), "engaging", target_pos=Position(x=7,y=7)),
        TurnLog(8, "S1", Position(x=3,y=3), "found", target_id="EN-S1", target_pos=Position(x=7,y=7)),
    ]

    payload = _make_payload_for_reports(logs)

    text = match_state_payload_to_text(payload)

    # Ensure each log type appears in the output
    assert "set a new target" in text
    assert "is returning to the carrier" in text
    assert "initiated an attack" in text
    assert "causing 5 damage" in text or "causing 5" in text
    assert "contact with S2 was lost" in text or "was lost" in text
    assert "landed on the carrier" in text
    assert "is engaging" in text
    assert "spotted" in text


def test_enemy_last_known_and_no_squadrons_message():
    # Uses top-level imports

    # Enemy intel with no squadrons visible
    payload = MatchStatePayload(
        match_id="m-3",
        status="active",
        turn=2,
        max_turn=20,
        waiting_for="none",
        map_w=10,
        map_h=10,
        units=SideViewPayload(carrier=None, squadrons=[]),
        intel=SideViewPayload(carrier=PayloadUnit(id="EN-C3", hp=60, max_hp=100, x=9, y=9, x0=8, y0=8), squadrons=[]),
        logs=[],
    )

    text = match_state_payload_to_text(payload)

    assert "Enemy intelligence:" in text
    assert "last known position" in text
    assert "- No enemy squadrons are currently visible." in text or "No enemy squadrons" in text


def test_carrier_target_and_squadron_states_and_stats():
    # Carrier has a target and fuel < 999 so 'operational radius' should appear.
    payload = MatchStatePayload(
        match_id="m-4",
        status="active",
        turn=4,
        max_turn=40,
        waiting_for="none",
        map_w=30,
        map_h=30,
        units=SideViewPayload(
            carrier=PayloadUnit(id="C3", hp=90, max_hp=100, x=10, y=10, fuel=50, speed=2, vision=4, target=Position(x=12, y=12)),
            squadrons=[
                PayloadUnit(id="S_out", hp=30, max_hp=40, x=11, y=11, state="outbound", fuel=8, speed=4, vision=5, target=Position(x=15, y=15)),
                PayloadUnit(id="S_eng", hp=20, max_hp=40, x=13, y=13, state="engaging", fuel=6, speed=4, vision=5, target=Position(x=16, y=16)),
                PayloadUnit(id="S_ret", hp=35, max_hp=40, x=9, y=9, state="returning", fuel=10, speed=4, vision=5),
                PayloadUnit(id="S_lost", hp=0, max_hp=40, x=8, y=8, state="lost", fuel=0, speed=4, vision=5),
            ],
        ),
        intel=SideViewPayload(carrier=None, squadrons=[]),
        logs=[],
    )

    text = match_state_payload_to_text(payload)

    assert "en route to target" in text
    assert "operational radius 50" in text
    assert "outbound toward to" in text or "outbound toward" in text
    assert "engaging an enemy target" in text
    assert "returning to the carrier" in text
    assert "annihilated" in text or "lost" in text


def test_result_message_when_finished():
    # When payload.result is set, header should indicate 'last battle' and show result description
    payload = MatchStatePayload(
        match_id="m-5",
        status="over",
        turn=10,
        max_turn=10,
        waiting_for="none",
        map_w=20,
        map_h=20,
        units=SideViewPayload(carrier=None, squadrons=[]),
        intel=SideViewPayload(carrier=PayloadUnit(id="EN-C4", hp=10, max_hp=100, x=5, y=5), squadrons=[]),
        logs=[],
        result="win",
    )

    text = match_state_payload_to_text(payload)

    assert "Below is the last battle situation." in text
    assert "Current result from your perspective: you have won." in text
    assert "Enemy intelligence:" in text
    assert "- Enemy carrier EN-C4" in text


def test_missing_carrier_and_unknown_target_in_logs():
    # No carrier present, and a log with unknown target should render 'unknown' language
    logs = [
        TurnLog(step=1, unit_id="Sx", unit_pos=None, report="found", target_id=None, target_pos=None),
    ]

    payload = MatchStatePayload(
        match_id="m-6",
        status="active",
        turn=5,
        max_turn=50,
        waiting_for="none",
        map_w=20,
        map_h=20,
        units=SideViewPayload(carrier=None, squadrons=[]),
        intel=SideViewPayload(carrier=None, squadrons=[]),
        logs=logs,
        result=None,
    )

    text = match_state_payload_to_text(payload)

    assert "- Your carrier position unknown." in text or "carrier position unknown" in text
    assert "spotted an unknown target" in text or "an unknown target" in text
