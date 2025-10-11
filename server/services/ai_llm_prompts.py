

# =========================
# MatchStatePayload helper
# =========================

from server.schemas import MatchStatePayload, PayloadUnit, Position, SideViewPayload, TurnLog

STATUS_DESCRIPTIONS: dict[str, str] = {
    "waiting": "waiting to begin",
    "active": "in progress",
    "over": "already finished",
}

WAITING_DESCRIPTIONS: dict[str, str] = {
    "none": "no pending actions",
    "orders": "awaiting system processing",
    "you": "waiting for your orders",
    "opponent": "waiting for the opponent's orders",
}

RESULT_DESCRIPTIONS: dict[str, str] = {
    "win": "you have won",
    "lose": "you have lost",
    "draw": "the battle ended in a draw",
}


def _format_xy(x: int|None, y: int|None) -> str:
    if x is None or y is None:
        return "unknown position"
    return f"({x}, {y})"


def _format_position_obj(pos: Position|None) -> str:
    if pos is None:
        return "unknown position"
    return _format_xy(pos.x, pos.y)


def _format_unit_stats(unit: PayloadUnit) -> list[str]:
    stats = [f"HP {unit.hp}/{unit.max_hp}"]
    if unit.fuel is not None and 0 <= unit.fuel < 999:
        stats.append(f"operational range {unit.fuel}")
    if unit.speed is not None:
        stats.append(f"speed {unit.speed}")
    if unit.vision is not None:
        stats.append(f"vision {unit.vision}")
    return stats

def _describe_carrier(label: str, carrier: PayloadUnit|None) -> str:
    if carrier is None:
        return f"- {label} carrier position unknown."
    details: list[str] = []
    if carrier.hp is None or carrier.hp<=0:
        details.append("sunk")
    else:
        if carrier.x is not None and carrier.y is not None:
            details.append(f"position {_format_xy(carrier.x, carrier.y)}")
        if carrier.target is not None:
            details.append(f"en route to target {_format_position_obj(carrier.target)}")
        else:
            details.append("stationary")
    details.extend(_format_unit_stats(carrier))
    detail_text = ", ".join(details)
    return f"- {label} carrier {carrier.id}: {detail_text}."

def _describe_squadrons(label: str, squadrons: list[PayloadUnit]) -> list[str]:
    lines: list[str] = []
    for squadron in squadrons:
        details: list[str] = []
        if squadron.state == 'onboard':
            details.append("on board a carrier, ready to launch.")
        elif squadron.state == 'lost':
            details.append("annihilated")
        elif squadron.state == 'outbound':
            details.append(f"position {_format_xy(squadron.x, squadron.y)}")
            details.append(f"outbound toward {_format_position_obj(squadron.target) if squadron.target else 'an assigned target'}")
        elif squadron.state == 'engaging':
            details.append(f"position {_format_xy(squadron.x, squadron.y)}")
            details.append(f"engaging an enemy target at {_format_position_obj(squadron.target) if squadron.target else 'an assigned target'}")
        elif squadron.state == 'returning':
            details.append(f"position {_format_xy(squadron.x, squadron.y)}")
            details.append("returning to the carrier")
        else:
            details.append("state unknown")
        details.extend(_format_unit_stats(squadron))
        lines.append(f"- {label} squadron {squadron.id}: {', '.join(details)}.")
    return lines

def _describe_side_view(label: str, view: SideViewPayload|None) -> list[str]:
    if view is None:
        return [f"- {label} information unavailable."]

    lines: list[str] = []
    carrier_line = _describe_carrier(label, view.carrier)
    if carrier_line:
        lines.append(carrier_line)

    squadron_list = view.squadrons or []
    if squadron_list:
        lines.extend(_describe_squadrons(label, squadron_list))

    return lines

def _describe_enemy_carrier(label: str, carrier: PayloadUnit|None) -> str:
    if carrier is None:
        return f"- {label} carrier position unknown."
    details: list[str] = []
    if carrier.hp is None or carrier.hp<=0:
        details.append("sunk")
    else:
        if carrier.x is not None and carrier.y is not None:
            if carrier.x0 is not None and carrier.y0 is not None:
                details.append(f"last known position {_format_xy(carrier.x, carrier.y)} (previously at {_format_xy(carrier.x0, carrier.y0)})")
            else:
                details.append(f"last known position {_format_xy(carrier.x, carrier.y)}")
        details.append(f"HP {carrier.hp}/{carrier.max_hp}")
    detail_text = ", ".join(details)
    return f"- {label} carrier {carrier.id}: {detail_text}."

def _describe_enemy_squadrons(label: str, squadrons: list[PayloadUnit]) -> list[str]:
    lines: list[str] = []
    for squadron in squadrons:
        details: list[str] = []
        if squadron.x is not None and squadron.y is not None:
            if squadron.x0 is not None and squadron.y0 is not None:
                details.append(f"last known position {_format_xy(squadron.x, squadron.y)} (previously at {_format_xy(squadron.x0, squadron.y0)})")
            else:
                details.append(f"last known position {_format_xy(squadron.x, squadron.y)}")
        details.append(f"HP {squadron.hp}/{squadron.max_hp}")
        lines.append(f"- {label} squadron {squadron.id}: {', '.join(details)}.")
    return lines

def _describe_enemy_view(label: str, view: SideViewPayload|None) -> list[str]:
    if view is None:
        return []

    lines: list[str] = []
    carrier_line = _describe_enemy_carrier(label, view.carrier)
    if carrier_line:
        lines.append(carrier_line)

    squadron_list = view.squadrons or []
    if squadron_list:
        lines.extend(_describe_enemy_squadrons(label, squadron_list))
    else:
        lines.append(f"- No {label.lower()} squadrons are currently visible.")

    return lines

def _enemy_unit_description( unit_id:str ) -> str:
    if unit_id.startswith("AC") or unit_id.startswith("BC"):
        return f"an enemy carrier({unit_id})"
    if unit_id.startswith("ASQ") or unit_id.startswith("BSQ"):
        return f"an enemy squadron({unit_id})"
    return f"an enemy unit({unit_id})"

def _format_target_description(log: TurnLog) -> str:
    if log.target_id and log.target_pos is not None:
        return f"{_enemy_unit_description(log.target_id)} at {_format_position_obj(log.target_pos)}"
    if log.target_id:
        return _enemy_unit_description(log.target_id)
    if log.target_pos is not None:
        return f"position {_format_position_obj(log.target_pos)}"
    return "an unknown target"

def _my_unit_description( unit_id:str ) -> str:
    if unit_id.startswith("AC") or unit_id.startswith("BC"):
        return f"your carrier({unit_id})"
    if unit_id.startswith("ASQ") or unit_id.startswith("BSQ"):
        return f"your squadron({unit_id})"
    return f"your unit({unit_id})"

def _describe_turn_log(log: TurnLog, recomment: bool) -> str:
    unit_location = ""
    if log.unit_pos is not None:
        unit_location = f" while at {_format_position_obj(log.unit_pos)}"

    target_description = _format_target_description(log)
    unit_description = _my_unit_description(log.unit_id)

    if log.report == "target":
        warning = ""
        if not log.target_id and (log.unit_id.startswith("ASQ") or log.unit_id.startswith("BSQ")):
            if log.target_pos and log.unit_pos and log.unit_pos.hex_distance(log.target_pos)<10:
                d = log.unit_pos.hex_distance(log.target_pos)
                if recomment and d<=15:  # 15マス以内は近すぎる
                    warning = f"  The distance is {d}. But unless the enemy carrier is right there, that's too close for reconnaissance. Should point further away."
        return f"{unit_description} set a new target toward {target_description}{unit_location}.{warning}"
    if log.report == "returning":
        return f"{unit_description} is returning to the carrier{unit_location}."
    if log.report == "attack":
        return f"{unit_description} initiated an attack on {target_description}{unit_location}."
    if log.report == "hit":
        damage = f" causing {log.value} damage" if log.value is not None else ""
        return f"{unit_description} hit {target_description}{damage}{unit_location}."
    if log.report == "lost":
        return f"contact with {unit_description} was lost{unit_location}."
    if log.report == "landed":
        return f"{unit_description} landed on the carrier{unit_location}."
    if log.report == "engaging":
        return f"{unit_description} is engaging {target_description}{unit_location}."
    if log.report == "found":
        return f"{unit_description} spotted {target_description}{unit_location}."
    return f"{unit_description} reported {log.report}{unit_location}."


def match_state_payload_to_text(payload: MatchStatePayload) -> str:
    """Convert a match snapshot into English text that is easy for an LLM to follow."""

    lines: list[str] = []

    if not payload.result:
        lines.append( "Below is the current battle situation from your perspective.")
    else:
        lines.append("Below is the last battle situation.")
        result_desc = RESULT_DESCRIPTIONS.get(payload.result, payload.result)
        lines.append(f"Result (from your perspective): {result_desc}.")

    lines.append("")
    lines.append(f"Turn {payload.turn} of {payload.max_turn}.")

    lines.append("")
    if payload.logs:
        lines.append("Recent events:")
        recomment:bool = payload.intel.carrier is None
        for log in payload.logs:
            lines.append(f"- {_describe_turn_log(log, recomment)}")

    if payload.result is None:
        if payload.intel.carrier is not None or (payload.intel.squadrons and len(payload.intel.squadrons)>0):
            lines.append("")
            lines.append("Enemy intelligence:")
            lines.extend(_describe_enemy_view("Enemy", payload.intel))

    lines.append("")
    lines.append("Your forces:")
    lines.extend(_describe_side_view("Your", payload.units))

    if payload.result is not None:
        lines.append("")
        lines.append("Enemy intelligence:")
        lines.extend(_describe_side_view("Enemy", payload.intel))

    lines.append("")

    return "\n".join(lines)
