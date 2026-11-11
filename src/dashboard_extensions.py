import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pydeck as pdk
import shap
import streamlit as st

from src.hotspot_analysis import load_incident_data, aggregate_hotspots
from src.xai import build_historical_features, calculate_shap
from src.pipeline import create_forecasting_features


@st.cache_data
def get_incident_data():
    """Load and cache historical crime incident data."""
    return load_incident_data()


def render_hotspot_tab(tab):
    """Render historical crime hotspot analysis."""

    with tab:
        st.header("Crime Hotspot Analysis")
        st.write(
            "Explore the geographical distribution of historical crime "
            "incidents using interactive filters."
        )

        try:
            df = get_incident_data()
        except Exception as e:
            st.error(f"Could not load crime incident data: {e}")
            return

        # Filters
        st.subheader("Filter Crime Records")

        col1, col2 = st.columns(2)

        min_year = int(df["YEAR"].min())
        max_year = int(df["YEAR"].max())

        with col1:
            year_range = st.slider(
                "Select year range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )

        with col2:
            months = sorted(df["MONTH"].unique().tolist())
            selected_months = st.multiselect(
                "Select months",
                options=months,
                default=months
            )

        col3, col4 = st.columns(2)

        with col3:
            neighbourhoods = sorted(
                df["NEIGHBOURHOOD"].dropna().unique().tolist()
            )
            selected_neighbourhoods = st.multiselect(
                "Select neighbourhoods",
                options=neighbourhoods,
                default=neighbourhoods
            )

        with col4:
            crime_types = sorted(df["TYPE"].dropna().unique().tolist())
            selected_types = st.multiselect(
                "Select crime categories",
                options=crime_types,
                default=crime_types
            )

        # Apply filters
        filtered = df[
            df["YEAR"].between(year_range[0], year_range[1])
            & df["MONTH"].isin(selected_months)
            & df["NEIGHBOURHOOD"].isin(selected_neighbourhoods)
            & df["TYPE"].isin(selected_types)
        ].copy()

        st.divider()

        if filtered.empty:
            st.warning("No crime records match the selected filters.")
            return

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Incidents", f"{len(filtered):,}")
        col2.metric(
            "Neighbourhoods",
            filtered["NEIGHBOURHOOD"].nunique()
        )
        col3.metric(
            "Crime Categories",
            filtered["TYPE"].nunique()
        )

        # Hotspot map
        st.subheader("Geographical Crime Hotspots")

        map_data = aggregate_hotspots(filtered)

        st.caption(
            "Each hotspot represents historical incidents grouped by "
            "rounded geographical coordinates. Brighter areas indicate "
            "higher incident concentration."
        )

        layer = pdk.Layer(
            "HeatmapLayer",
            data=map_data,
            get_position="[Longitude, Latitude]",
            get_weight="Incident_Count",
            radiusPixels=40,
            intensity=1,
            threshold=0.05,
            pickable=True
        )

        view_state = pdk.ViewState(
            latitude=float(filtered["Latitude"].mean()),
            longitude=float(filtered["Longitude"].mean()),
            zoom=10,
            pitch=0
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "Incidents: {Incident_Count}"}
        )

        st.pydeck_chart(deck, use_container_width=True)

        # Top neighbourhoods
        st.subheader("Top 10 Neighbourhoods by Crime Count")

        top_neighbourhoods = (
            filtered.groupby("NEIGHBOURHOOD")
            .size()
            .reset_index(name="Incident_Count")
            .sort_values("Incident_Count", ascending=False)
            .head(10)
        )

        fig = px.bar(
            top_neighbourhoods.sort_values(
                "Incident_Count", ascending=True
            ),
            x="Incident_Count",
            y="NEIGHBOURHOOD",
            orientation="h",
            title="Historical Crime Distribution",
            labels={
                "Incident_Count": "Number of Incidents",
                "NEIGHBOURHOOD": "Neighbourhood"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Neighbourhood Crime Summary")
        st.dataframe(
            top_neighbourhoods,
            use_container_width=True,
            hide_index=True
        )


def render_xai_tab(tab, model, history, metadata):
    """Render SHAP-based explainable AI analysis."""

    with tab:
        st.header("SHAP Explainability")
        st.write(
            "Understand how historical crime patterns influence the "
            "XGBoost model's monthly crime forecast."
        )

        try:
            feature_names = metadata["feature_names"]

            X, explanation = calculate_shap(
                model,
                build_historical_features(history, feature_names)
            )

        except Exception as e:
            st.error(f"Unable to calculate SHAP explanations: {e}")
            return

        # Global feature importance
        st.subheader("Global Feature Importance")

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Mean Absolute SHAP": abs(explanation.values).mean(axis=0)
        }).sort_values("Mean Absolute SHAP", ascending=True)

        fig = px.bar(
            importance,
            x="Mean Absolute SHAP",
            y="Feature",
            orientation="h",
            title="Impact of Features on Model Predictions",
            labels={
                "Mean Absolute SHAP": "Mean Absolute SHAP Value",
                "Feature": "Model Feature"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Higher mean absolute SHAP values indicate features that "
            "have a greater influence on model predictions."
        )

        # SHAP summary plot
        st.divider()
        st.subheader("SHAP Summary Plot")

        st.write(
            "This plot shows how different feature values influence "
            "the model's predictions across historical observations."
        )

        try:
            fig_shap, ax = plt.subplots(figsize=(10, 6))

            shap.summary_plot(
                explanation.values,
                X,
                show=False
            )

            st.pyplot(fig_shap, clear_figure=True)
            plt.close(fig_shap)

        except Exception as e:
            st.warning(f"Could not generate SHAP summary plot: {e}")

        # Individual forecast explanation
        st.divider()
        st.subheader("Individual Forecast Explanation")

        st.write(
            "The following explanation examines the model's latest "
            "one-step monthly forecast using the most recent historical data."
        )

        try:
            latest_features = create_forecasting_features(history)
            latest_features = latest_features.reindex(
                columns=feature_names
            )

            explainer = shap.TreeExplainer(model)
            latest_explanation = explainer(latest_features)

            predicted_value = float(model.predict(latest_features)[0])

            st.metric(
                "Predicted Monthly Crime Count",
                f"{predicted_value:,.0f}"
            )

            fig_waterfall, ax = plt.subplots(figsize=(10, 6))

            shap.plots.waterfall(
                latest_explanation[0],
                show=False
            )

            st.pyplot(fig_waterfall, clear_figure=True)
            plt.close(fig_waterfall)

            contribution_table = pd.DataFrame({
                "Feature": feature_names,
                "Feature Value": latest_features.iloc[0].values,
                "SHAP Contribution": latest_explanation.values[0]
            })

            contribution_table["Absolute Contribution"] = (
                contribution_table["SHAP Contribution"].abs()
            )

            contribution_table = contribution_table.sort_values(
                "Absolute Contribution",
                ascending=False
            )

            st.subheader("Feature Contribution Details")

            st.dataframe(
                contribution_table,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:
            st.warning(f"Could not generate individual explanation: {e}")

        st.info(
            "Important: SHAP explains how the model uses its input "
            "features. It does not establish real-world causes of crime. "
            "These explanations concern monthly crime-count forecasts, "
            "not individual offenders or future geographical hotspots."
        )
