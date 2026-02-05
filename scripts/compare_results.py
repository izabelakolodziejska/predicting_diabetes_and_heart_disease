import json
import pandas as pd

def compare_results():
    
    with open('results/single_task_diabetes/results.json', encoding='utf-8') as f:
        stl_diab = json.load(f)
    
    with open('results/single_task_heart/results.json', encoding='utf-8') as f:
        stl_heart = json.load(f)
    
    with open('results/multi_task/results.json', encoding='utf-8') as f:
        mtl = json.load(f)
    
    
    comparison = pd.DataFrame({
        'Metryka': ['AUC-ROC', 'F1-score', 'Precision', 'Recall', 'Accuracy'],
        'STL-Cukrzyca': [
            stl_diab['auc'],
            stl_diab['f1'],
            stl_diab['precision'],
            stl_diab['recall'],
            stl_diab['accuracy']
        ],
        'MTL-Cukrzyca': [
            mtl['diabetes']['metrics']['auc'],
            mtl['diabetes']['metrics']['f1'],
            mtl['diabetes']['metrics']['precision'],
            mtl['diabetes']['metrics']['recall'],
            mtl['diabetes']['metrics']['accuracy']
        ],
        'STL-Choroby serca':  [
            stl_heart['auc'],
            stl_heart['f1'],
            stl_heart['precision'],
            stl_heart['recall'],
            stl_heart['accuracy']
        ],
        'MTL-Choroby serca': [
            mtl['heart']['metrics']['auc'],
            mtl['heart']['metrics']['f1'],
            mtl['heart']['metrics']['precision'],
            mtl['heart']['metrics']['recall'],
            mtl['heart']['metrics']['accuracy']
        ]
    })
    
    
    comparison['$\\Delta$ Cukrzyca (\\%)'] = (  # ← POPRAWIONE
        (comparison['MTL-Cukrzyca'] - comparison['STL-Cukrzyca']) / 
        comparison['STL-Cukrzyca'] * 100
    ).round(2)
    
    comparison['$\\Delta$ Choroby serca (\\%)'] = (  # ← POPRAWIONE
        (comparison['MTL-Choroby serca'] - comparison['STL-Choroby serca']) / 
        comparison['STL-Choroby serca'] * 100
    ).round(2)
    
    print("\n" + "="*80)
    print("STL vs MTL COMPARISON")
    print("="*80)
    print(comparison.to_string(index=False))
    
    
    latex_tabular = comparison.to_latex(
        index=False, 
        float_format="%.4f", 
        escape=False  
    )
    
    
    latex_full = f"""\\begin{{table}}[h]
\\centering
\\caption{{Porównanie wyników STL vs MTL}}
\\label{{tab:stl_mtl_comparison}}
{latex_tabular}
\\end{{table}}
"""
    
    
    with open('results/comparison_table.tex', 'w', encoding='utf-8') as f:
        f.write(latex_full)
    
    print("\n✓ Saved to results/comparison_table.tex")

if __name__ == "__main__":
    compare_results()