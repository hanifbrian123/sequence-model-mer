import json
import sys

def parse_notebook(ipynb_file, txt_file):
    with open(ipynb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    with open(txt_file, 'w', encoding='utf-8') as f:
        for i, cell in enumerate(nb.get('cells', [])):
            cell_type = cell.get('cell_type', '')
            f.write(f"\n{'='*60}\n")
            f.write(f"CELL {i} [{cell_type.upper()}]\n")
            f.write(f"{'-'*60}\n")
            
            source = "".join(cell.get('source', []))
            f.write(source)
            f.write("\n")
            
            if cell_type == 'code' and cell.get('outputs'):
                f.write(f"\n{'-'*30} OUTPUT {'-'*30}\n")
                for out in cell['outputs']:
                    if out.get('output_type') == 'stream':
                        f.write("".join(out.get('text', [])))
                    elif out.get('output_type') == 'execute_result' or out.get('output_type') == 'display_data':
                        data = out.get('data', {})
                        if 'text/plain' in data:
                            f.write("".join(data['text/plain']))
                            f.write("\n")
                f.write(f"\n{'-'*68}\n")

if __name__ == '__main__':
    parse_notebook('scratch/external_notebook.ipynb', 'scratch/nb_with_outputs.txt')
    print("Extracted to scratch/nb_with_outputs.txt")
