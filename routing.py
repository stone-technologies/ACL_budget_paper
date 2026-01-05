import pandas as pd
from scipy import stats

def apply_routing(results_path, B1=512, B2=1024, out_path=None):
    if isinstance(results_path, str):
        results_df = pd.read_csv(results_path)
    else:
        results_df = results_path
    
    routed = []
    for (dataset, budget, unit), group in results_df.groupby(['dataset', 'budget', 'unitization']):
        if budget <= B1:
            pick = group[group['selector'] == 'lead']
        elif budget <= B2:
            pick = group[group['selector'] == 'mmr']
        else:
            pick = group[group['selector'] == 'facility']
        
        if len(pick) > 0:
            for _, row in pick.iterrows():
                row_dict = row.to_dict()
                row_dict['routed'] = True
                routed.append(row_dict)
    
    routed_df = pd.DataFrame(routed)
    
    print(f"Routing: Lead (B<={B1}), MMR ({B1}<B<={B2}), Facility (B>{B2})")
    print("-" * 60)
    
    def mean_se(x):
        mean = x.mean()
        se = stats.sem(x) if len(x) > 1 else 0
        return f"{mean:.3f}±{se:.3f}"
    
    for dataset in routed_df['dataset'].unique():
        subset = routed_df[routed_df['dataset'] == dataset]
        print(f"\n{dataset.upper()} (Routed)")
        
        pivot = subset.pivot_table(
            values='r1',
            index='unitization',
            columns='budget',
            aggfunc=mean_se
        )
        print(pivot.to_string())
        
        if 'r1_llm' in subset.columns and subset['r1_llm'].notna().any():
            print(f"\n{dataset.upper()} (Routed, LLM)")
            pivot_llm = subset.pivot_table(
                values='r1_llm',
                index='unitization',
                columns='budget',
                aggfunc=mean_se
            )
            print(pivot_llm.to_string())
    
    if out_path:
        routed_df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")
    
    return routed_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--B1', type=int, default=512)
    parser.add_argument('--B2', type=int, default=1024)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    
    if args.out is None:
        args.out = args.input.replace('.csv', '_routed.csv')
    
    apply_routing(args.input, B1=args.B1, B2=args.B2, out_path=args.out)
