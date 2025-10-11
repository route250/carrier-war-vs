import sys,os
import json
import csv

if __name__ == '__main__':
    # ai_gemini.py is at server/services; project root is two levels up
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.services.ai_llm_base import json_loads
from tests.bench_llm_competition import OUTPUT_DIR
import re


def build_anchor_id(comp_id: str, col_index: int) -> str:
    """Generate a stable, HTML-safe anchor id for the given competition column."""
    base = re.sub(r"[^A-Za-z0-9_-]", "-", comp_id.strip()) if comp_id else ""
    if not base:
        base = f"col-{col_index}"
    return f"{base}-col-{col_index}"


def main():

    """
     OUTPUT_DIRが以下の構造になっている。
     OUTPUT_DIR/{competition_id}/数字/{model_name}_log.json

     OUTPUT_DIR以下をスキャンして、_log.jsonをパースして、
    """

    histories:dict[str,list[list[str]]] = {}
    pair_map: dict[str, tuple[str, str]] = {}
    for competition_id in os.listdir(OUTPUT_DIR):
        competition_dir = os.path.join(OUTPUT_DIR, competition_id)
        if not os.path.isdir(competition_dir):
            continue
        for entry in os.listdir(competition_dir):
            entry_dir = os.path.join(competition_dir, entry)
            if not os.path.isdir(entry_dir):
                continue
            if not entry.isdigit():
                continue
            pair=[]
            for filename in os.listdir(entry_dir):
                if not filename.endswith('_log.json'):
                    continue
                model_name = filename[:-9]  # remove '_log.json'
                pair.append(model_name)
            if len(pair) != 2:
                print(f"Skipping {entry_dir} because it does not contain exactly two _log.json files")
                continue
            pair_map[competition_id] = (pair[0], pair[1])

            for filename in os.listdir(entry_dir):
                if not filename.endswith('_log.json'):
                    continue
                model_name = filename[:-9]  # remove '_log.json'
                log_path = os.path.join(entry_dir, filename)            
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                if not isinstance(log_data, list):
                    continue
                msgs:list[str] = [competition_id]
                while len(msgs) < 32:
                    msgs.append("")  # Prepare for up to 31 turns + competition_id
                turn_num = 0
                last_state = ""
                for m in log_data:
                    if not isinstance(m, dict):
                        continue
                    content = m.get("content","")
                    if not content:
                        continue
                    if "assistant" !=m.get("role",""):
                        last_state = content
                        # Below is the current battle situation from your perspective.\n\nTurn 2 of 30.\n\n
                        m_turn = re.search(r"^Below is the current battle situation from your perspective\..*?Turn (\d+)", content, re.DOTALL)
                        if m_turn:
                            num = int(m_turn.group(1))
                            if num != turn_num+1:
                                raise ValueError(f"Turn number jump in {log_path}: {turn_num} -> {num}")
                            turn_num = num
                        continue

                    text = ""
                    try:
                        mm = None
                        mm = json_loads(content)
                        if not isinstance(mm, dict):
                            continue
                        thinking = mm.get("thinking","")
                        if "Turn 30" in thinking:
                            print(f"Skipping Turn 30 entry in {log_path}")
                        act = mm.get("action",{})
                        if isinstance(act, dict):
                            order = json.dumps(act, ensure_ascii=False)
                        else:
                            order = json.dumps({
                                "carrier_target": mm.get("carrier_target"),
                                "launch_target": mm.get("launch_target"),
                            }, ensure_ascii=False)
                        text = thinking.strip()+"\n```json\n"+order+"\n```"
                    except Exception as e:
                        print(f"Error parsing JSON from {log_path}: {e}")
                        text = content.strip()

                    pretext = msgs[turn_num]
                    if pretext:
                        msgs[turn_num] = pretext + "\n\n" + text
                    else:
                        msgs[turn_num] = text
                if last_state:
                    msgs[turn_num+1] = last_state
                if msgs:
                    histories.setdefault(model_name, []).append(msgs)

    for model_name, rows0 in histories.items():
        rows = []
        nn = max([len(r) for r in rows0])
        for i in range(nn):
            aa = []
            for row in rows0:
                if i < len(row):
                    aa.append(row[i])
                else:
                    aa.append("")  # Fill missing entries with empty strings
            rows.append(aa)
        csv_path = os.path.join(OUTPUT_DIR, f'hist_{model_name}.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as fp:
            writer = csv.writer(fp)
            writer.writerows(rows)
        print(f"Results saved to {csv_path}")
    
        html_path = os.path.join(OUTPUT_DIR, f'hist_{model_name}.html')
        with open(html_path, 'w', encoding='utf-8') as fp:
            fp.write('<!doctype html>\n')
            fp.write('<html><head><meta charset="utf-8"><style>\n')
            fp.write('table{table-layout:fixed;border-collapse:collapse;width:100%;}\n')
            fp.write('td{width:400px;border:1px solid #ccc;padding:10px;white-space:pre-wrap;word-break:break-word;overflow-wrap:anywhere;vertical-align:top;}\n')
            fp.write('th{background:#f7f7f7;border:1px solid #ccc;padding:10px;vertical-align:top;}\n')
            # Column header color (the top row headers) and row header color (first column headers)
            fp.write('.col-header{background:#e6f0ff;}\n')
            fp.write('.row-header{background:#fff4e6;}\n')
            fp.write('</style></head><body>\n')
            # Insert model name at the top of the HTML for clarity
            fp.write(f'<h2>Model: {model_name}</h2>\n')
            # create table and explicit colgroup so widths are respected
            if rows:
                num_cols = len(rows[0])
            else:
                num_cols = 0
            fp.write('<table>\n')
            if num_cols:
                fp.write('<colgroup>\n')
                fp.write('<col style="width:50px">\n')
                for _ in range(num_cols):
                    fp.write('<col style="width:400px">\n')
                fp.write('</colgroup>\n')
            for row_idx, row in enumerate(rows[0:1]):
                fp.write('<tr>\n')
                fp.write(f'<th class="row-header">turn</th>\n')
                # The first element of each column header row is competition_id (rows[0][0])
                for col_index, cell in enumerate(row):
                    comp_id = cell.strip() if isinstance(cell, str) else ""
                    anchor_id = comp_id
                    # Determine the opponent model using pair_map if available
                    other_model = None
                    try:
                        pair = pair_map.get(comp_id)
                        if pair:
                            # pair is a tuple (a,b). pick the one that is not the current model_name
                            if pair[0] == model_name:
                                other_model = pair[1]
                            elif pair[1] == model_name:
                                other_model = pair[0]
                    except Exception:
                        other_model = None

                    if not other_model:
                        # fallback to self-link when opponent not found
                        target_file = f'hist_{model_name}.html'
                    else:
                        target_file = f'hist_{other_model}.html'

                    fp.write(f'<th id="{anchor_id}" class="col-header">\n')
                    fp.write(f'<a href="{target_file}#{anchor_id}">')
                    fp.write(cell.replace('\n', '<br>'))
                    fp.write('</a>\n')
                    fp.write('</th>\n')
                fp.write('</tr>\n')
            
            for r, row in enumerate(rows[1:]):
                fp.write('<tr>\n')
                fp.write(f'<th class="row-header">{r+1}</th>\n')
                for cell in row:
                    fp.write('<td>\n')
                    fp.write(cell.replace('\n', '<br>') + '\n')
                    fp.write('</td>\n')
                fp.write('</tr>\n')
            fp.write('</table>\n')
            fp.write('</body></html>\n')
        print(f"Results saved to {html_path}")

if __name__ == '__main__':
    main()
