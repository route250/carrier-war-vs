

# =========================
# MatchStatePayload helper
# =========================
from typing import Literal
from pydantic import BaseModel, Field, constr, ValidationError
from server.schemas import SQUADRON_RANGE, MatchStatePayload, PayloadUnit, Position, SideViewPayload, TurnLog
from server.services.hexmap import rawmap_to_text


# 役割/基本方針
SYSTEM_PROMPT: str = (
    "海戦型SLGの指揮官として自軍ユニットを操作してゲームをプレイし勝利を目指して下さい。\n"
    "ゲームのルールから勝利条件を満たすための作戦を計画立案し実行し常に最善のプランへ修正し勝利を目指して下さい。\n"
)
SYSTEM_PROMPT_RULE:str = (
    "# ルール\n"
    " - 空母(carrier):\n"
    "     海上のみ移動可能。航空部隊の発着艦と対空戦闘のみ可能。空母へは攻撃できない。"
    "     {enemy_location}\n"
	" - 航空部隊(squadron):\n"
    "     launch_targetに目標座標を指定してorderすると発艦して目標座標へ向かう。\n"
    "     launch_targetは空母を中心に距離{Range}ヘクス以内に指定できる。\n"
    "     行動中は帰投まで目標変更できない。\n"
	" - 索敵:\n"
    "     空母・航空部隊は索敵範囲の敵を発見可能(移動経路含む)\n"
    "     航空部隊は目標地点に到達した後、敵空母を発見できなければ帰還に移る。索敵範囲はvisionの範囲のみ\n"
	" - 戦闘:\n"
	"     航空部隊 vs 空母のみ（航空部隊同士は戦闘しない）\n"
    "     航空部隊は航空部隊に攻撃できない。\n"
    "     航空部隊は往路に空母を発見すると攻撃に移る。復路では攻撃せず報告だけする。\n"
    "     ゲーム中にHPは回復しない。\n"
	" - 勝敗:\n"
	"     敵空母撃沈または敵航空部隊全滅で勝ち\n"
	"     {max_turn}ターン終了 → HP多い方勝ち。HPが同じなら先に敵空母を発見した方が勝ち\n"
    "# 作戦のヒント\n"
    " - 敵空母を発見する為には、航空部隊を発艦して索敵させる必要があります。可能な限り遠方で敵空母を発見するための行動を取りましょう。\n"
    " - 空母同士の航空戦では、相手よりも先に敵空母を発見できれば圧倒的有利です。\n"
    "   航空機の航続距離は限られているため、索敵したい海域に少し空母を近づける必要があります。ただし近づけ過ぎると敵に発見されるリスクが高まります。\n"
    " - 敵に発見・攻撃された場合は、波状攻撃から逃れるために敵に見つからない位置を予想して空母を移動させる。\n"
    "   また、敵機が北方向や帰還していく方向から敵空母の位置を推定することも出来るはずです。\n"
    " - 敵空母を先に発見できなければ、ほぼ負け確定です。"
)
DESCRIPTION_THINKING = "(必須)敵空母が存在する海域、存在しない海域の推定。索敵結果から推測を構築。作戦プランを作成更新。"
DESCRIPTION_CARRIER_TARGET = "空母(carrier)の移動目標を指示すると目的地に向かって進み続ける。nullは変更なし"
DESCRIPTION_LAUNCH_TARGET = (
    "発艦指示。航空部隊(squadron)に対して索敵・攻撃の目標座標を指定し発艦させる。nullは指示なし。"
    " (Specify the target coordinates for reconnaissance/attack for the squadron and launch. null means no instruction.)"
)
#
#空母の移動目標（または null）
#航空部隊の索敵・攻撃の目標位置（または null）
# 出力フォーマットの厳密指定
# SYSTEM_PROMPT_OUTPUT_FORMAT: str = (
#     "出力は必ず以下のJSONオブジェクトのみで余計な説明やコードブロックは書かないでください。\n"
#     f"thinking(必須):{DESCRIPTION_THINKING}\n"
#     f"carrier_target:{DESCRIPTION_CARRIER_TARGET}\n"
#     f"launch_target:{DESCRIPTION_LAUNCH_TARGET}\n"
#     "{\n"
#     "  \"thinking\": \"索敵プラン、敵空母の居る海域、存在しない海域の推定、発艦指示\",\n"
#     "  \"action\": {\n"
#     "    \"carrier_target\": {\"x\": <int>, \"y\": <int>} | null,\n"
#     "    \"launch_target\": {\"x\": 目標X, \"y\": 目標Y} | null\n"
#     "  }\n"
#     "}\n"
# )

# 制約条件
SYSTEM_PROMPT_CONSTRAINTS: str = (
    "- 空母を移動する時は、carrier_target に座標を指定する。\n"
    "- 航空部隊(onboard状態)を発艦する時は、launch_target に目標座標を指定する。\n"
    "- thinking は必ず書くこと。分析や戦術、記録などを簡潔に述べる。"
)

# 基本ナレッジ
def default_knowledge( language:Literal["ja", "en"]|None ) -> str:
    if language == "en":
        return (
            "# Knowledge\n"
            " - Launch air units to scout for the enemy, rather than waiting to discover them.\n"
            " - It is crucial to predict and estimate the location of the enemy carrier and discover it as early and as far away as possible.\n"
            " - Use the movements of enemy aircraft from reconnaissance information to predict the location of the enemy carrier and move your carrier within the operational range of your air units.\n"
            " - Move your carrier to avoid being discovered by predicting the routes of enemy air units.\n"
        )
    else:
        return (
            "# ナレッジ\n"
            " - 敵を発見してからではなく、敵を発見するために航空部隊を飛ばしましょう。\n"
            " - 敵空母の位置を予測・推定し、できるだけ早く遠くで敵空母を発見することが重要です。\n"
            " - 索敵情報の敵航空機の動きを活用して、敵空母の位置を予想し航空部隊の航続範囲まで空母を移動しましょう。\n"
            " - 敵航空部隊の進路を予想して空母が発見されないように空母を移動しましょう。\n"
        )


# 要約プロンプト
def build_summary_prompt( language:Literal["ja", "en"]|None ) -> str:
    if language == "en":
        return (
            "Summarize the current battle situation up to this point, including information necessary for future operations, in the <think> tag. "
            "Since this is just a summary, please set the <carrier_target> and <launch_target> tags to null.\n"
        )
    else:
        return (
            "現時点までの作戦状況について、今後の作戦に必要な情報を短く要約して<think>タグに記述して下さい。今回は要約だけなので<carrier_target>タグと<launch_target>タグはnullにして下さい。\n"
        )


REVIEW_PROMPT: str = (
    "作戦終了です。今回の作戦をレビューして、失敗、成功、注意点などをまとめて。ナレッジの内容を書き直して下さい。\n"
    "次の作戦へのLLMプロンプトとして短くまとめて下さい。\n"
    "あなたが理解できればいいので出来るだけ短く。あなたが理解できるなら人間の言葉でなくてもよく出来るだけ短く\n"
)


class Coordinate(BaseModel):
    x: int = Field(..., description="目標x座標")
    y: int = Field(..., description="目標y座標")

class Action(BaseModel):
    carrier_target: Coordinate|None = Field(None, description=f"{DESCRIPTION_CARRIER_TARGET}")
    launch_target: Coordinate|None = Field(None, description=f"{DESCRIPTION_LAUNCH_TARGET}")

    @staticmethod
    def to_json_format() -> str:
        fmt = "{"
        fmt += f" \"carrier_target\": {{\"x\": <int>, \"y\": <int>}} | null,"
        fmt += f" \"launch_target\": {{\"x\": <int>, \"y\": <int>}} | null"
        fmt += "}"
        return fmt

class ResponseModel(BaseModel):
    thinking: str = Field(..., description=f"{DESCRIPTION_THINKING}")
    action: Action = Field(..., description="CarrierとSquadronの移動先座標、偵察先座標、攻撃目標の座標を指示する")

    @staticmethod
    def to_json_format() -> str:
        fmt = "{"
        fmt += f" \"thinking\": \"{DESCRIPTION_THINKING}\","
        fmt += f" \"action\": {Action.to_json_format()}"
        fmt += "}"
        return fmt


def build_system_prompt( grid:list[list[int]], side:Literal["A", "B"]|None, max_turn:int, language:Literal["ja", "en"] ) -> str:
    """分割定数を結合した最終 SYSTEM_PROMPT を返す。"""
    prompt_list = [SYSTEM_PROMPT]

    enemy_location = ""
    if side == "A":
        enemy_location = "敵空母の初期位置はマップの右下(26,26)近傍ランダム位置です。"
    elif side == "B":
        enemy_location = "敵空母の初期位置はマップの左上(3,3)近傍のランダム位置です。"
    r = SQUADRON_RANGE
    rule = SYSTEM_PROMPT_RULE.format( enemy_location=enemy_location, Range=r, max_turn=max_turn)
    prompt_list.append(rule)

    map_content = build_map_content(grid, language)
    if map_content and len(map_content) > 0:
        prompt_list.append(map_content)

    # if self._knowledge_content and len(self._knowledge_content) > 0:
    #     prompt_list.append(self._knowledge_content)
    # else:
    #     prompt_list.append(BASE_KNOWLEDGE)

    prompt_list.append(f"制約:\n{SYSTEM_PROMPT_CONSTRAINTS}")

    return "\n\n".join(prompt_list)

def build_map_content(grid:list[list[int]], language:Literal["ja", "en"]|None) -> str:
    """ヘクスマップ情報を JSON 文字列で返す。"""
    try:
        msg = "hexamap" if language == "en" else "ヘクスマップ"
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0

        if grid and w > 0 and h > 0:
            xmap = []
            xmap.append(f"{msg} width={w} height={h} legend(0=sea,1=land)")
            xmap.append("```")
            for line in rawmap_to_text(grid):
                xmap.append(line)
            xmap.append("```")
            return "\n".join(xmap)
    except Exception as ex:
        print(f"build_map_content error: {ex}")
        pass
    return ""


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
