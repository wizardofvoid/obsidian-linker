from main import app

def main():
    print("Generating graph visualization for Obsidian Linker agent...")
    
    mmd_path = "graph_structure.mmd"
    png_path = "graph_visualization.png"
    
    try:
        # Get graph representation
        graph = app.get_graph()
        
        # 1. Save Mermaid markdown representation
        mermaid_code = graph.draw_mermaid()
        with open(mmd_path, "w", encoding="utf-8") as f:
            f.write(mermaid_code)
        print(f"  [1/2] Saved Mermaid Markdown structure to: {mmd_path}")
        
        # 2. Save PNG representation using LangGraph's native rendering
        png_bytes = graph.draw_mermaid_png()
        with open(png_path, "wb") as f:
            f.write(png_bytes)
        print(f"  [2/2] Saved PNG graph visualization image to: {png_path}")
        print("\nSuccess! Open 'graph_visualization.png' to view your agent's workflow graph.")
        
    except Exception as e:
        print(f"\nError generating PNG visualization: {e}")
        print("Note: PNG generation may require internet access to fetch from the Mermaid API, or optional dependencies.")
        print(f"However, the Mermaid structure was successfully written to '{mmd_path}'.")
        print("You can copy its contents and paste them into https://mermaid.live to view and edit the graph online!")

if __name__ == "__main__":
    main()
