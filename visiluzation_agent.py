import re
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from typing import Dict, List
import plotly.figure_factory as ff

class Visualization:
    """Generates suitable visualizations based on query results."""

    def __init__(self, groq_api_key):
        self.llm = ChatGroq(model="llama3-70b-8192", temperature=0, groq_api_key=groq_api_key)

    def recommend_visualization(self, df: pd.DataFrame, query: str, result) -> Dict:
        """Suggests best visualization types based on query and result."""
        
        viz_prompt = PromptTemplate.from_template(
            """You are a data visualization expert. Based on the given query and result, suggest the best visualization types.

            Query: {query}
            Data Sample:
            {sample}

            Use the following visualization types when applicable:
            - "bar" (For categorical vs. numerical data)
            - "scatter" (For small datasets and relationships)
            - "pie" (For percentage distributions)
            - "histogram" (For numerical distributions)
            - "box" (For statistical summaries)
            - "violin" (For probability density comparisons)
            - "density_heatmap" (For correlation analysis)
            - "kde" (For smooth distributions)
            - "area" (For trends over time)
            - "line_area" (For date-based trends with shaded areas)

            Ensure your response is a valid JSON with up to 4 visualization recommendations.
            Each visualization should include:
            - "type": The chart type (bar, scatter, pie, line, heatmap, etc.)
            - "data_columns": The relevant DataFrame columns (must match column names exactly)
            - "title": A meaningful title

            Only use column names from this list:
            {columns}

            Example Output:
            ```json
            {{
                "recommendations": [
                    {{"type": "bar", "data_columns": ["Category", "Sales"], "title": "Sales by Category"}},
                    {{"type": "scatter", "data_columns": ["Profit", "Discount"], "title": "Profit vs Discount"}}
                ]
            }}
            ```"""
        )

        sample_str = df.head(4).to_string()
        viz_message = viz_prompt.format_prompt(query=query, sample=sample_str, columns=list(df.columns))
        viz_response = self.llm.invoke([HumanMessage(content=viz_message.to_string())])
        viz_content = viz_response.content.strip()

        match = re.search(r"```json(.*?)```", viz_content, re.DOTALL)
        if match:
            viz_content = match.group(1).strip()

        try:
            recommendations = json.loads(viz_content)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response from LLM", "raw_output": viz_content}

        return recommendations
    
    def auto_generate_visualizations(self, df: pd.DataFrame) -> Dict:
        """You are a Power BI and Tableau expert and Dynamically generates suitable visualizations based on dataset structure."""
    
        recommendations = {"recommendations": []}
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

        # ✅ 1. Time-Series Trends (Date-Based Visualizations)
        if datetime_cols and numeric_cols:
            recommendations["recommendations"].append(
                {"type": "line_area", "data_columns": [datetime_cols[0], numeric_cols[0]], "title": f"Trend of {numeric_cols[0]} Over Time"}
            )
    
        # ✅ 2. Categorical Data (Bar Charts)
        if categorical_cols and numeric_cols:
            recommendations["recommendations"].append(
                {"type": "bar", "data_columns": [categorical_cols[0], numeric_cols[0]], "title": f"{numeric_cols[0]} by {categorical_cols[0]}"}
            )

        # ✅ 3. Numerical Relationship (Scatter Plot for large datasets, Density Heatmap for correlation)
        if len(numeric_cols) >= 2:
            if df.shape[0] > 1000:  # Large dataset → Use density heatmap
                recommendations["recommendations"].append(
                    {"type": "density_heatmap", "data_columns": [numeric_cols[0], numeric_cols[1]], "title": f"Density Heatmap of {numeric_cols[0]} vs {numeric_cols[1]}"}
                )
            else:  # Small dataset → Use scatter plot
                recommendations["recommendations"].append(
                    {"type": "scatter", "data_columns": [numeric_cols[0], numeric_cols[1]], "title": f"{numeric_cols[0]} vs {numeric_cols[1]}"}
                )

        # ✅ 4. Distribution Analysis (Histograms & KDE Plots)
        if numeric_cols:
            recommendations["recommendations"].append(
                {"type": "histogram", "data_columns": [numeric_cols[0]], "title": f"Histogram of {numeric_cols[0]}"}
            )
            recommendations["recommendations"].append(
                {"type": "kde", "data_columns": [numeric_cols[0]], "title": f"KDE Distribution of {numeric_cols[0]}"}
            )

        # ✅ 5. Box Plot (For Outlier Detection)
        if len(numeric_cols) > 0:
            recommendations["recommendations"].append(
                {"type": "box", "data_columns": [numeric_cols[0]], "title": f"Box Plot of {numeric_cols[0]}"}
            )

        # ✅ 6. Violin Plot (For Skewed Distributions)
        if len(numeric_cols) > 0:
            recommendations["recommendations"].append(
                {"type": "violin", "data_columns": [numeric_cols[0]], "title": f"Violin Plot of {numeric_cols[0]}"}
            )

        return recommendations


    def generate_visualization(self, df: pd.DataFrame, recommendations: List[Dict]) -> List:
        """Generates visualizations from recommendations."""
        visualizations = []
        
        # Custom color palettes
        color_palettes = {
            "bar": "Blues",
            "scatter": "Viridis",
            "histogram": "Cividis",
            "box": "Plasma",
            "violin": "ocean",
            "density_heatmap": "Turbo",
            "kde": "Inferno",
            "area": "Cividis",
            "line_area": "Cividis"
        }

        for rec in recommendations.get("recommendations", []):
            viz_type = rec.get("type", "").lower()
            data_columns = rec.get("data_columns", [])
            title = rec.get("title", "Generated Visualization")

            try:
                color = color_palettes.get(viz_type, "Blues")  # Default color palette

                if viz_type == "bar":
                    fig = px.bar(df, x=data_columns[0], y=data_columns[1], title=title, color=data_columns[0], color_continuous_scale=color)
                elif viz_type == "scatter":
                    fig = px.scatter(df, x=data_columns[0], y=data_columns[1], title=title, color=data_columns[0], color_continuous_scale=color)
                elif viz_type == "pie":
                    fig = px.pie(df, names=data_columns[0], title=title)  # Pie charts use default color
                elif viz_type == "box":
                    fig = px.box(df, y=data_columns[0], title=title, color_discrete_sequence=px.colors.sequential.Plasma)
                elif viz_type == "violin":
                    fig = px.violin(df, y=data_columns[0], title=title, box=True, points="all", color_discrete_sequence=px.colors.sequential.Magma)
                elif viz_type == "density_heatmap":
                    fig = px.density_heatmap(df, x=data_columns[0], y=data_columns[1], title=title, color_continuous_scale=color)
                elif viz_type == "kde":
                    fig = ff.create_distplot([df[data_columns[0]].dropna()], group_labels=[data_columns[0]], show_hist=False)
                elif viz_type == "area":
                    fig = px.area(df, x=data_columns[0], y=data_columns[1], title=title, color=data_columns[0], color_continuous_scale=color)
                elif viz_type == "line_area":
                    fig = px.area(df, x=data_columns[0], y=data_columns[1], title=title, color=data_columns[0], color_continuous_scale=color)
                else:
                    continue

                visualizations.append(fig)
            except Exception as e:
                print(f"⚠️ Error generating {viz_type}: {e}")

        return visualizations
