import json
from src.utils.data import load_product_data
from src.agent.router import llm_route
from src.agent.compare_tool import build_structured_comparison
from src.retriever.multi_source import multi_source_retrieval
from src.agent.explainer import llm_explain
from src.retriever.structured import SPEC_MAP

product_df, raw_df = load_product_data()
print("产品数量:", len(product_df))
print("评论数量:", len(raw_df))
print(product_df['review_count'].describe())


def main():
    product_df, _ = load_product_data()

    query = input("Enter your query: ").strip()
    providers = "openai"

    routing = llm_route(query, providers=providers)
    print("\n[ROUTER OUTPUT]\n", routing)

    top = multi_source_retrieval(product_df, routing, query, top_k=5, candidate_k=40)
    if top.empty:
        print("\nNo products match your constraints. Try increasing budget or removing strict spec constraints.")
        return

    print("\n[TOP PRODUCTS]\n")
    cols = ["brand", "model", "price_usd", "rating"] + list(SPEC_MAP.values())
    for extra_col in ["multi_source_score", "structured_signal_score", "review_relevance_score", "description_relevance_score"]:
        if extra_col in top.columns:
            cols.append(extra_col)
    print(top[cols])

    compare_result = None
    if str(routing.get("intent") or "").upper() == "COMPARE" and len(top) >= 2:
        compare_result = build_structured_comparison(top, routing, query=query, max_items=3)
        print("\n[STRUCTURED COMPARE TOOL]\n")
        print(json.dumps(compare_result, ensure_ascii=False, indent=2))

    explanation = llm_explain(query, routing, top, providers=providers, compare_result=compare_result)
    print("\n[LLM EXPLANATION]\n")
    print(explanation)


    product_df, _ = load_product_data()
    print(len(product_df))  # 产品数量
    print(product_df['review_count'].describe())  # 每个产品平均多少评论



if __name__ == "__main__":
    main()
