#!/usr/bin/env python3
"""
Kp Index Space Weather Monitor

A monitoring system that tracks the Kp geomagnetic index from GFZ Potsdam
and sends automated email alerts when space weather conditions exceed specified thresholds.

Data Source: GFZ German Research Centre for Geosciences

"""

import logging
import re
import shutil
import smtplib
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

# Display times in CET (Europe/Berlin handles CET/CEST)
CET = ZoneInfo("Europe/Berlin")

import markdown
import numpy as np
import pandas as pd
import requests
import typer

from src.config import MonitorConfig


@dataclass
class AnalysisResults:
    """
    AnalysisResults containing analysis results with keys:

    Parameters
    ----------
    max_kp : float
        Maximum Kp value in current forecast
    threshold_exceeded: bool
        Boolean indicating if threshold exceeded
    high_kp_records : pd.DataFrame
        Records above alert threshold
    next_24h_forecast : pd.DataFrame
        Forecast for next 24 hours
    alert_worthy : bool
        Boolean indicating if alert should be sent
    probability_df : pd.DataFrame
        DataFrame containing probability of Kp exceeding threshold
    storm_prob_df : pd.DataFrame
        DataFrame containing storm probabilities (prob 6-7, prob 7-8, prob >= 8)
    """

    max_kp: float
    max_df: pd.Series
    threshold_exceeded: bool
    high_kp_records: pd.DataFrame
    next_24h_forecast: pd.DataFrame
    alert_worthy: bool
    probability_df: pd.DataFrame
    storm_prob_df: pd.DataFrame

    def __getitem__(self, key):
        return getattr(self, key)


# fmt: off
KP_TO_DECIMAL = {
    "0": 0.00, "0+": 0.33,
    "1-": 0.67, "1": 1.00, "1+": 1.33,
    "2-": 1.67, "2": 2.00, "2+": 2.33,
    "3-": 2.67, "3": 3.00, "3+": 3.33,
    "4-": 3.67, "4": 4.00, "4+": 4.33,
    "5-": 4.67, "5": 5.00, "5+": 5.33,
    "6-": 5.67, "6": 6.00, "6+": 6.33,
    "7-": 6.67, "7": 7.00, "7+": 7.33,
    "8-": 7.67, "8": 8.00, "8+": 8.33,
    "9-": 8.67, "9": 9.00
}
# fmt: on
DECIMAL_TO_KP = {v: k for k, v in KP_TO_DECIMAL.items()}


class KpMonitor:
    """
    Main monitoring class for Kp index space weather data.

    Handles data fetching, analysis, alerting, and email notifications
    for geomagnetic activity monitoring.
    """

    IMAGE_PATH = "./assets/kp_swift_ensemble_LAST.png"
    IMAGE_PATH_SWPC = "./assets/kp_swift_ensemble_with_swpc_LAST.png"
    CSV_PATH = "./assets/kp_product_file_SWIFT_LAST.csv"
    VIDEO_PATH_AURORA = "./assets/aurora_forecast.mp4"
    IMAGE_PATH_AURORA = "./assets/aurora_LAST.png"

    # Caption for the forecast plot (SWPC + Min-Max)
    FORECAST_IMAGE_CAPTION = "Bar colours indicate geomagnetic activity levels: green corresponds to quiet conditions (Kp &lt; 3), yellow to moderate activity (3 &lt; Kp &le; 6), and red to high storm conditions (Kp &gt; 6). The red dashed line shows the official NOAA SWPC Kp forecast. Error bars represent the minimum-maximum spread of forecast Kp values."

    def __init__(self, config: MonitorConfig, log_suffix: str = "") -> None:
        self.last_alert_time = None
        self.last_max_kp = 0
        self.config = config
        self.log_folder = Path(self.config.log_folder)
        self.debug_with_swpc = self.config.debug_with_swpc
        self.log_folder.mkdir(parents=True, exist_ok=True)
        self.config.kp_alert_threshold = np.round(self.config.kp_alert_threshold, 2)
        self.kp_threshold_str = DECIMAL_TO_KP[self.config.kp_alert_threshold]
        self.LOCAL_IMAGE_PATH = self.copy_image()
        self.LOCAL_AURORA_VIDEO_PATH = None  # set when building message with AURORA WATCH
        self.current_utc_time = pd.Timestamp(datetime.now(timezone.utc))
        self.log_suffix = log_suffix
        self.setup_logging()

    def copy_image(self) -> str:
        """
        Copies the appropriate Kp forecast image to the current directory.

        Returns
        -------
        str
            Path to the copied image file.
        """
        if self.debug_with_swpc:
            return shutil.copy2(self.IMAGE_PATH_SWPC, "./kp_swift_ensemble_with_swpc_LAST.png")
        return shutil.copy2(self.IMAGE_PATH, "./kp_swift_ensemble_LAST.png")

    def copy_aurora_video(self) -> str:
        """Copy the aurora video to the current directory for html embedding."""
        return shutil.copy2(self.VIDEO_PATH_AURORA, "./aurora_forecast.mp4")

    def copy_aurora_image(self) -> str:
        """Copy the Kp-dependent auroral intensity image to the current directory for html embedding."""
        return shutil.copy2(self.IMAGE_PATH_AURORA, "./aurora_LAST.png")

    def setup_logging(self) -> None:
        """
        Configure logging to file and console.

        Sets up logging handlers for both file and console output with
        appropriate formatting and log levels from configuration.
        """

        def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        sys.excepthook = log_uncaught_exceptions

        logging.basicConfig(
            level=self.config.log_level,
            format="[%(levelname)-8s] %(asctime)s - %(name)s:%(lineno)d - %(message)s",
            handlers=[
                logging.FileHandler(
                    self.log_folder
                    / f"kp_monitor_{self.log_suffix}_{datetime.now(timezone.utc).strftime('%Y%d%m')}.log"
                ),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def fetch_kp_data(self, test: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch current Kp index forecast data from GFZ website.

        Parameters
        ----------
        test : bool, optional
            If True, create a test DataFrame with synthetic data instead of reading from CSV, by default False

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing forecast data or None if fetch fails
        """
        if not test:
            try:
                df = pd.read_csv(self.CSV_PATH)

                df["Time (UTC)"] = pd.to_datetime(df["Time (UTC)"], format="%d-%m-%Y %H:%M", dayfirst=True, utc=True)
                df.index = df["Time (UTC)"]
                self.logger.info(f"Successfully fetched {len(df)} records")
                return df

            except pd.errors.EmptyDataError:
                self.logger.error("Received empty CSV file")
                return None
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}", exc_info=True)
                return None
        else:
            self.logger.info("Running in test mode - generating synthetic Kp data")
            return self.get_test_data()

    def round_to_step(self, x, step=0.33):
        """Round value to nearest step (e.g. Kp index steps)."""
        return np.round(x / step) * step

    def get_test_data(self) -> pd.DataFrame:
        """Generate synthetic Kp forecast data for testing without real CSV."""
        time_index = pd.date_range(start=self.current_utc_time, periods=10, freq="3h")

        data = {"Time (UTC)": time_index}

        kp_members = []

        for i in range(20):
            vals = np.round(np.random.choice(np.linspace(0, 9, 28), len(time_index)), 2)
            data[f"kp_{i}"] = vals
            kp_members.append(vals)

        kp_array = np.vstack(kp_members).T

        data["minimum"] = np.min(kp_array, axis=1)
        data["0.25-quantile"] = np.quantile(kp_array, 0.25, axis=1)
        data["median"] = np.median(kp_array, axis=1)
        data["0.75-quantile"] = np.quantile(kp_array, 0.75, axis=1)
        data["maximum"] = np.max(kp_array, axis=1)

        data["prob 4-5"] = np.mean((kp_array >= 4) & (kp_array < 5), axis=1)
        data["prob 5-6"] = np.mean((kp_array >= 5) & (kp_array < 6), axis=1)
        data["prob 6-7"] = np.mean((kp_array >= 6) & (kp_array < 7), axis=1)
        data["prob 7-8"] = np.mean((kp_array >= 7) & (kp_array < 8), axis=1)
        data["prob >= 8"] = np.mean(kp_array >= 8, axis=1)

        df = pd.DataFrame(data)
        df.index = df["Time (UTC)"]

        return df

    def analyze_kp_data(self, df: pd.DataFrame) -> AnalysisResults:
        """
        Analyze Kp forecast data for alert conditions.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing Kp forecast data from GFZ

        Returns
        -------
        `AnalysisResults`
            `AnalysisResults` containing analysis results with keys as described in the `AnalysisResults` dataclass
        """
        try:
            # Get current maximum values
            self.logger.info(f"Current UTC Time: {self.current_utc_time}")
            max_values = df[df.index >= self.current_utc_time]["maximum"]

            if max_values.empty or np.isnan(max_values.max()):
                # No forecast entries at / after current time – treat as no alert case
                self.logger.warning("No forecast data at or after current time; skipping alert analysis.")
                probability = pd.Series(0.0, index=df.index, name="Probability")
                probability_df = pd.DataFrame({"Time (UTC)": df["Time (UTC)"], "Probability": probability})
                probability_df.index = probability_df["Time (UTC)"]
                probability_df.drop(columns=["Time (UTC)"], inplace=True)

                empty = df.iloc[0:0].copy()
                storm_prob_df = pd.DataFrame(
                    columns=["Time (UTC)", "Probability 6-7", "Probability 7-8", "Probability >= 8"]
                )
                return AnalysisResults(
                    max_kp=float("nan"),
                    max_df=max_values,
                    threshold_exceeded=False,
                    high_kp_records=empty,
                    next_24h_forecast=empty,
                    alert_worthy=False,
                    probability_df=probability_df.round(2),
                    storm_prob_df=storm_prob_df,
                )

            max: float = np.round(max_values.max(), 2)

            self.ensembles = [col for col in df.columns if re.match(r"kp_\d+", col)]
            self.total_ensembles = len(self.ensembles)
            probability = np.sum(df[self.ensembles] >= self.config.kp_alert_threshold, axis=1) / self.total_ensembles
            high_kp_records = df[df["maximum"].astype(float) >= self.config.kp_alert_threshold].copy()
            high_kp_records = high_kp_records[high_kp_records["Time (UTC)"] >= self.current_utc_time].copy()
            next_24h = df[df["Time (UTC)"] >= self.current_utc_time].head(9).copy()

            high_kp_records["Time (UTC)"] = pd.to_datetime(high_kp_records["Time (UTC)"], utc=True)
            next_24h["Time (UTC)"] = pd.to_datetime(next_24h["Time (UTC)"], utc=True)

            probability_df = pd.DataFrame({"Time (UTC)": df["Time (UTC)"], "Probability": probability})
            probability_df.index = probability_df["Time (UTC)"]
            probability_df.drop(columns=["Time (UTC)"], inplace=True)
            probability_df = probability_df.replace({"Probability": {1.0: 0.95}})

            storm_prob_df = df[df.index >= self.current_utc_time][["prob 6-7", "prob 7-8", "prob >= 8"]].copy()

            analysis = AnalysisResults(
                max_kp=max,
                max_df=max_values,
                threshold_exceeded=max > self.config.kp_alert_threshold,
                high_kp_records=high_kp_records.round(2),
                next_24h_forecast=next_24h.round(2),
                alert_worthy=len(high_kp_records) > 0,
                probability_df=probability_df.round(2),
                storm_prob_df=storm_prob_df,
            )

            if not np.isnan(max):
                self.logger.info(
                    f"Analysis complete - Current Kp: {DECIMAL_TO_KP[max]}, Alert: {analysis['alert_worthy']}, Threshold: {self.kp_threshold_str}"
                )
            else:
                self.logger.info(
                    f"Analysis complete - Current Kp: nan, Alert: {analysis['alert_worthy']}, Threshold: {self.kp_threshold_str}"
                )
            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing data: {e}", exc_info=True)
            empty = df.iloc[0:0].copy()
            probability = pd.Series(0.0, index=df.index, name="Probability")
            probability_df = pd.DataFrame({"Time (UTC)": df["Time (UTC)"], "Probability": probability})
            probability_df.index = probability_df["Time (UTC)"]
            probability_df.drop(columns=["Time (UTC)"], inplace=True)
            storm_prob_df = pd.DataFrame(columns=["Time (UTC)", "Prob Kp 6-7", "Prob 7-8", "Prob Kp >= 8"])
            return AnalysisResults(
                max_kp=float("nan"),
                max_df=pd.Series(dtype=float),
                threshold_exceeded=False,
                high_kp_records=empty,
                next_24h_forecast=empty,
                alert_worthy=False,
                probability_df=probability_df.round(2),
                storm_prob_df=storm_prob_df,
            )

    def footer(self) -> str:
        return f"""
*This is an automated alert from the Kp Index Monitoring System using GFZ Space Weather Forecast.*

---

<small>
© {datetime.now().year} GFZ Helmholtz Centre for Geosciences | GFZ Helmholtz-Zentrum für Geoforschung  
The data/data products are provided "as-is" without warranty of any kind either expressed or implied, including but not limited to the implied warranties of merchantability, correctness and fitness for a particular purpose. The entire risk as to the quality and performance of the Data/data products is with the Licensee.
In no event will GFZ be liable for any damages direct, indirect, incidental, or consequential, including damages for any lost profits, lost savings, or other incidental or consequential damages arising out of the use or inability to use the data/data products.
</small>
            """

    def _kp_html_table(self, record: pd.DataFrame, probabilities: pd.DataFrame) -> str:
        """Generate markdown table for Kp index records."""
        table = f"""
| Time (CET) | Probability (Kp ≥ {self.kp_threshold_str}) | Min Kp Index<sup>[<a href="#fn1">1</a>]</sup> | Max Kp Index<sup>[<a href="#fn2">2</a>]</sup> | Median Kp Index<sup>[<a href="#fn3">3</a>]</sup> | Activity<sup>[<a href="#fn4">4</a>][<a href="#fn5">5</a>]</sup> |
|------------|-------------------------------------------|------------------|------------------|---------------------|------------------|
"""
        for _, row in record.iterrows():
            kp_val_max = np.round(row["maximum"], 2)
            kp_val_med = np.round(row["median"], 2)
            kp_val_min = np.round(row["minimum"], 2)
            _, level_min, color_min = self.get_status_level_color(kp_val_min)
            _, level_max, color_max = self.get_status_level_color(kp_val_max)

            time_idx = row["Time (UTC)"]
            prob = probabilities.loc[time_idx, "Probability"]

            time_str = row["Time (UTC)"].tz_convert(CET).strftime("%d.%m.%Y %H:%M")
            prob_str = f"{prob * 100:.0f}%"
            activity_str = f'<span style="color: {color_min};">{level_min}</span> - <span style="color: {color_max};">{level_max}</span>'

            table += f"| **{time_str}** | **{prob_str}** | **{DECIMAL_TO_KP[kp_val_min]}** | **{DECIMAL_TO_KP[kp_val_max]}** | **{DECIMAL_TO_KP[kp_val_med]}** | {activity_str} |\n"

        table += """
<a id="fn1"></a><sup>1</sup> Min Kp Index: Minimum value of Kp Ensembles  
<a id="fn2"></a><sup>2</sup> Max Kp Index: Maximum value of Kp Ensembles  
<a id="fn3"></a><sup>3</sup> Median Kp Index: Median value of Kp Ensembles  
<a id="fn4"></a><sup>4</sup> Geomagnetic Activity Level based on Min-Max range
"""
        return table

    def _storm_probability_table(self, storm_prob_df: pd.DataFrame) -> str:
        """Generate markdown table for storm probabilities (Kp 6-7, 7-8, >=8)."""
        if storm_prob_df.empty:
            return ""

        filtered_rows = []
        for idx, row in storm_prob_df.iterrows():
            total_prob = row["prob 6-7"] + row["prob 7-8"] + row["prob >= 8"]
            if total_prob > 0.50:
                filtered_rows.append((idx, row))

        if not filtered_rows:
            return ""

        table = """
## STORM PROBABILITY FORECAST (Kp ≥ 6)

| Time (CET) | Probability of kp 6-7 | Probability of kp 7-8 | Probability of kp ≥8 |
|------------|----------|----------|---------|
"""
        for idx, row in filtered_rows:
            time_cet = idx.tz_convert(CET).strftime("%d.%m.%Y %H:%M")
            prob_6_7 = f"{row['prob 6-7'] * 100:.0f}%"
            prob_7_8 = f"{row['prob 7-8'] * 100:.0f}%"
            prob_8_plus = f"{row['prob >= 8'] * 100:.0f}%"
            table += f"| {time_cet} | {prob_6_7} | {prob_7_8} | {prob_8_plus} |\n"

        table += "\n*Showing timestamps where combined probability of Kp ≥ 6 exceeds 50%*\n"
        return table

    def get_observed_kp(self, start: pd.Timestamp) -> Tuple[str, float] | None:
        """
        Fetch observed Kp index data from GFZ API.

        Parameters
        ----------
        start : pd.Timestamp
            Start time for fetching observed Kp data

        Returns
        -------
        Tuple[str, float] or None
            Tuple with datetime and Kp value if found, None otherwise

        """
        try:
            max_attempts = 8  # search back up to 24 hours (8 * 3-hou)
            attempts = 0

            while attempts < max_attempts:
                start_date_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
                end_date_str = (start + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
                url = f"https://kp.gfz.de/app/json/?start={start_date_str}&end={end_date_str}&index=Kp"

                self.logger.info(f"Fetching observed Kp data from {start_date_str} to {end_date_str}")

                response = requests.get(url)
                response.raise_for_status()

                data = response.json()

                if len(data.get("Kp", [])) > 0:
                    self.logger.info(f"Observed Kp data found for {data['datetime'][0]} : {data['Kp'][0]}")
                    return data["datetime"][0], data["Kp"][0]

                else:
                    self.logger.warning(f"No observed Kp data found for {start_date_str}, shifting 3 hours back")
                    start -= timedelta(hours=3)
                    attempts += 1

            self.logger.warning("No observed Kp data found after multiple shifts")
            return None

        except Exception as e:
            self.logger.error(f"Error fetching observed Kp data: {e}", exc_info=True)
            return None

    def create_message(self, analysis: AnalysisResults) -> str:
        """
        Create formatted alert message for high Kp conditions using Markdown.

        Parameters
        ----------
        analysis : `AnalysisResults`
            `AnalysisResults` containing analysis results from analyze_kp_data

        Returns
        -------
        message : str
            Formatted Markdown alert message
        """
        high_records = analysis["high_kp_records"]
        probability_df = analysis["probability_df"]
        probability_df = probability_df[probability_df.index >= self.current_utc_time]
        current_kp = analysis.next_24h_forecast["median"].iloc[0]
        status, _, _ = self.get_status_level_color(current_kp)

        max_values = analysis["max_df"]

        prob_at_time = 24  # hours
        target_time = self.current_utc_time + pd.Timedelta(hours=prob_at_time)
        nearest_idx = target_time.round("3h")
        high_prob_value = probability_df[:nearest_idx]["Probability"].max()

        threshold_status, threshold_level, _ = self.get_status_level_color(self.config.kp_alert_threshold)

        max_kp_at_finite_time = np.round(max_values.max(), 2)

        max_kp_at_finite_time_status, max_kp_at_finite_time_level, _ = self.get_status_level_color(
            max_kp_at_finite_time
        )
        mask = probability_df["Probability"] >= 0.4
        if mask.any():
            start_time = probability_df.index[mask][0]
            end_time = probability_df.index[mask][-1]
        else:
            start_time = high_records["minimum"].idxmax()
            end_time = high_records["maximum"].idxmax()

        observed_time, observed_kp = self.get_observed_kp(analysis.next_24h_forecast.index[0])
        prob_at_start_time = probability_df.loc[start_time]["Probability"]

        if observed_kp is not None:
            observed_status, _, _ = self.get_status_level_color(observed_kp)
        else:
            observed_status = "DATA NOT AVAILABLE YET"

        start_time_kp_min_status, _, _ = self.get_status_level_color(high_records.loc[start_time]["minimum"].min())
        end_time_kp_max_status, _, _ = self.get_status_level_color(high_records.loc[end_time]["maximum"].max())

        start_cet = start_time.tz_convert(CET)
        end_cet = end_time.tz_convert(CET)
        if start_time == end_time:
            message_prefix = f"""At {start_cet.strftime("%H:%M (CET) %d.%m.%Y")} """
        else:
            message_prefix = (
                f"""From {start_cet.strftime("%H:%M (CET) %d.%m.%Y")}  to {end_cet.strftime("%H:%M (CET) %d.%m.%Y")}"""
            )
        if observed_time != analysis.next_24h_forecast.index[0]:
            obs_utc = datetime.strptime(observed_time.strip(), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            obs_message_prefix = (
                f""" (Observed Kp data available up to {obs_utc.astimezone(CET).strftime("%H:%M CET %d.%m.%Y")})"""
            )
        else:
            obs_message_prefix = ""

        # Use "=" for Kp 9 (maximum value), "≥" for all other values
        kp_comparison = "=" if max_kp_at_finite_time == 9 else "≥"

        message = f"""<h2 style="color: #d9534f;">SPACE WEATHER ALERT - {end_time_kp_max_status} ({max_kp_at_finite_time_level}) with probability ≥ {prob_at_start_time * 100:.0f}% predicted</h2>


### {message_prefix}, space weather can reach {end_time_kp_max_status} with Kp {kp_comparison} {DECIMAL_TO_KP[max_kp_at_finite_time]} with probability ≥ {prob_at_start_time * 100:.0f}%.

**Current Conditions:** {observed_status.replace("CONDITIONS", "")} {obs_message_prefix}

**Alert sent at:** {datetime.now(timezone.utc).astimezone(CET).strftime("%H:%M CET %d.%m.%Y ")}

![Forecast Image](cid:forecast_image)

<p class="forecast-caption">{self.FORECAST_IMAGE_CAPTION}</p>


"""
        # message += self._kp_html_table(high_records, probability_df)

        # Add storm probability table if there are rows with combined prob > 50%
        storm_table = self._storm_probability_table(analysis["storm_prob_df"])
        if storm_table:
            message += storm_table
            message += "\n"

        # First add the geomagnetic activity scale section
        message += """## GEOMAGNETIC ACTIVITY SCALE"""
        message += self.get_storm_level_description_table()
        message += "\n"

        # Then optionally add the aurora watch section
        AURORA_KP = 7
        high_records_above_threshold = high_records[
            (high_records["minimum"].astype(float) >= AURORA_KP)
            | (high_records["median"].astype(float) >= AURORA_KP)
            | (high_records["maximum"].astype(float) >= AURORA_KP)
        ]

        if not high_records_above_threshold.empty:
            message += f"""
## **AURORA WATCH:**

<div class="aurora-watch-row">
  <div class="aurora-col">
    <img src="aurora_LAST.png" alt="Kp-dependent Auroral Intensity Prediction" class="aurora-forecast-img" />
  </div>
  <div class="aurora-col">
    <video class="aurora-forecast-video" controls autoplay loop muted>
      <source src="aurora_forecast.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
</div>

**Note:** Kp ≥ {DECIMAL_TO_KP[AURORA_KP]} indicate potential auroral activity at Berlin latitudes. Time indicated in UTC.

"""

        message += self.footer()

        return message.strip()

    def create_subject(self, analysis: AnalysisResults) -> str:
        """
        Create email subject line based on analysis results.

        Parameters
        ----------
        analysis : `AnalysisResults`
            `AnalysisResults` containing analysis results from analyze_kp_data

        Returns
        -------
        subject : str
            Email subject line
        """
        _, level_min, _ = self.get_status_level_color(analysis["high_kp_records"]["minimum"].max())
        _, level_max, _ = self.get_status_level_color(analysis["high_kp_records"]["maximum"].max())

        subject = f"Predicted Geomagnetic Activity from {level_min} - {level_max}"
        return subject.strip()

    def get_storm_level_description_table(self) -> str:
        """Generate markdown table for geomagnetic storm levels."""
        G1 = "[NOAA [G1]](https://www.swpc.noaa.gov/noaa-scales-explanation#:~:text=G%201)"
        G2 = "[NOAA [G2]](https://www.swpc.noaa.gov/noaa-scales-explanation#:~:text=G%202)"
        G3 = "[NOAA [G3]](https://www.swpc.noaa.gov/noaa-scales-explanation#:~:text=G%203)"
        G4 = "[NOAA [G4]](https://www.swpc.noaa.gov/noaa-scales-explanation#:~:text=G%204)"
        G5 = "[NOAA [G5]](https://www.swpc.noaa.gov/noaa-scales-explanation#:~:text=G%205)"

        rows = [
            ("Quiet", "0-3", "Quiet conditions"),
            ("Active", "4", "Moderate geomagnetic activity"),
            ("Minor Storm (G1)", "5", f"Weak power grid fluctuations. For more details see {G1}"),
            ("Moderate Storm (G2)", "6", f"High-latitude power systems affected. For more details see {G2}"),
            ("Strong Storm (G3)", "7", f"Power systems may need voltage corrections. For more details see {G3}"),
            ("Severe Storm (G4)", "8", f"Possible widespread voltage control problems. For more details see {G4}"),
            ("Extreme Storm (G5)", "9", f"Widespread power system voltage control problems. For more details see {G5}"),
        ]

        table = """
| Level | Kp Value | Description |
|-------|----------|-------------|
"""
        for level, kp_value, desc in rows:
            table += f"| **{level}** | **{kp_value}** | {desc} |\n"

        return table

    def get_status_level_color(self, kp: float) -> tuple[str, str, str]:
        """Get geomagnetic status, level, and color based on Kp value.

        Parameters
        ----------
        kp : float
            Kp index value

        Returns
        -------
        status : str
            Geomagnetic activity status description
        level : str
            Geomagnetic storm level (e.g., [G1], [G2], etc.)
        color : str
            Hex color code representing severity
        """
        status = "UNKNOWN"
        level = "[?]"
        color = "#000000"
        if kp == 9:
            status = "EXTREME STORM CONDITIONS"
            level = "G5"
            color = "#FE0004"
        elif kp >= 8:
            status = "SEVERE STORM CONDITIONS"
            level = "G4"
            color = "#FE0004"
        elif kp >= 7:
            status = "STRONG STORM CONDITIONS"
            level = "G3"
            color = "#FD0007"
        elif kp >= 6:
            status = "MODERATE STORM CONDITIONS"
            level = "G2"
            color = "#FF4612"
        elif kp >= 5:
            status = "MINOR STORM CONDITIONS"
            level = "G1"
            color = "#FE801D"
        elif kp >= 4:
            status = "MODERATE CONDITIONS"
            level = "MODERATE"
            color = "#FFFA3D"
        else:
            status = "QUIET CONDITIONS"
            level = "QUIET"
            color = "#5cb85c"
        return status, level, color

    def send_alert(self, subject: str, message: str) -> bool:
        """
        Send email using the system's configured SMTP (without calling `mail`).

        Parameters
        ----------
        subject : str
            Email subject line
        message : str
            Email message content (HTML formatted)

        Returns
        -------
        bool
            True if email sent successfully, False otherwise
        """
        try:
            recipients = self.config.recipients
            self.construct_and_send_email(recipients, subject, message)

            self.logger.info(f"Mail sent successfully to {len(recipients)} recipients")
            return True

        except Exception as e:
            self.logger.error(f"Error sending mail: {e}", exc_info=True)
            return False

    def should_send_alert(self, analysis: AnalysisResults) -> bool:
        """
        Determine if alert should be sent to avoid spam.

        Parameters
        ----------
        analysis : AnalysisResults
            AnalysisResults containing analysis results from analyze_kp_data

        Returns
        -------
        bool
            True if alert should be sent, False otherwise
        """
        if not analysis["alert_worthy"]:
            return False
        current_time = pd.Timestamp.now(tz="UTC")
        if self.last_alert_time:
            time_since_last_alert = (current_time - self.last_alert_time).total_seconds() / 3600
            if time_since_last_alert < 6:
                self.logger.warning("Skipping alert - too soon since last alert")
                return False

        return True

    def run_single_check(self, test: bool = False) -> bool:
        """
        Execute a single monitoring check cycle.

        Fetches Kp data, analyzes it, and sends alerts if necessary.

        Parameters
        ----------
        test : bool, optional
            If True, runs in test mode with synthetic data, by default False

        Returns
        -------
        bool
            True if check completed successfully, False otherwise
        """
        self.logger.info("Kp Index check")
        df = self.fetch_kp_data(test=test)
        if df is None:
            return False
        analysis = self.analyze_kp_data(df)

        if self.should_send_alert(analysis):
            max_kp = analysis["max_kp"]

            message = self.create_message(analysis)
            subject = self.create_subject(analysis)

            _ = self.copy_image()
            self.LOCAL_AURORA_VIDEO_PATH = self.copy_aurora_video()
            _ = self.copy_aurora_image()
            email_sent = self.send_alert(subject, message)
            message_for_file = markdown.markdown(
                message.replace("cid:forecast_image", self.LOCAL_IMAGE_PATH),
                extensions=["tables", "fenced_code", "footnotes", "nl2br"],
            )
            html_output = self.basic_html_format(message_for_file)
            with open("index.html", "w") as f:
                f.write(html_output)

            if email_sent:
                self.last_alert_time = pd.Timestamp.now(tz="UTC")
                self.last_max_kp = max_kp
        else:
            self.logger.info(
                f"No alert needed - Current Kp: {analysis['max_kp']:.2f}, Threshold: {self.kp_threshold_str}"
            )

        return True

    def construct_and_send_email(self, recipients: list[str], subject: str, message: str) -> None:
        """Construct and send an email with HTML content and embedded image.

        Parameters
        ----------
        recipients : list[str]
            List of recipient email addresses
        subject : str
            Email subject line
        message : str
            Email message content (Markdown formatted, will be converted to HTML)
        """
        html_message = markdown.markdown(
            message,
            extensions=[
                "tables",
                "fenced_code",
                "footnotes",
                "nl2br",
            ],
        )
        html_message = self.basic_html_format(html_message)

        # root message as multipart/related
        msg_root = MIMEMultipart("related")
        msg_root["From"] = "pager"
        msg_root["Reply-To"] = "jhawar@gfz-potsdam.de"

        if len(recipients) == 1:
            msg_root["To"] = recipients[0]
        else:
            msg_root["Bcc"] = ", ".join(recipients)
        msg_root["Subject"] = subject

        msg_alternative = MIMEMultipart("alternative")
        msg_root.attach(msg_alternative)

        plain_text = "Your email client does not support HTML."
        msg_alternative.attach(MIMEText(plain_text, "plain"))

        msg_alternative.attach(MIMEText(html_message, "html"))
        with open(self.LOCAL_IMAGE_PATH, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<forecast_image>")
            img.add_header("Content-Disposition", "inline", filename="forecast_image.png")
            msg_root.attach(img)

        # Note: Video attachments are not well supported in email clients
        # The video will be available in the generated HTML file

        with smtplib.SMTP("localhost") as smtp:
            smtp.send_message(msg_root)

    def basic_html_format(self, message: str) -> str:
        formatting = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6; color: #000000; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                    table {{ border-collapse: collapse; margin: 20px 0; width: 100%; font-size: 13px; }}
                    th, td {{ padding: 8px 12px; border: 1px solid #ddd; text-align: left; }}
                    th {{ background-color: #f0f0f0; font-weight: bold; }}
                    img {{ max-width: 100%; height: auto; }}
                    h1 {{ color: #d9534f; font-size: 1.5rem; }}
                    h2 {{ color: #5bc0de; margin-top: 30px; font-size: 1.25rem; }}
                    h3 {{ color: #000000; font-size: 1.1rem; }}
                    .forecast-caption {{ font-size: 16px; color: #444444; margin-top: 4px; }}
                    small {{ font-size: 11px; color: #333; }}
                    hr {{ border: 0; border-top: 1px solid #ddd; margin: 20px 0; }}
                    .aurora-watch-row {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 16px 0; }}
                    .aurora-col {{ flex: 1 1 0; min-width: 280px; }}
                    .aurora-forecast-img {{ width: 100%; height: auto; display: block; }}
                    .aurora-forecast-video {{ width: 100%; height: auto; display: block; }}
                </style>
            </head>
            <body>
            {message}
            </body>
            </html>
            """
        return formatting

    def run_continuous_monitoring(self) -> None:
        """
        Run continuous monitoring with specified check intervals.

        Runs indefinitely, checking Kp data at configured intervals and
        sending alerts when thresholds are exceeded. Can be stopped with
        Ctrl+C (KeyboardInterrupt).
        """
        self.logger.info("Starting continuous Kp index monitoring")
        self.logger.info(f"Check interval: {self.config.check_interval_hours} hours")
        self.logger.info(f"Alert threshold: {self.config.kp_alert_threshold}")

        while True:
            try:
                self.run_single_check(test=False)

                # Wait for next check
                sleep_seconds = self.config.check_interval_hours * 3600
                self.logger.info(f"Waiting {self.config.check_interval_hours} hours until next check...")
                time.sleep(sleep_seconds)

            except KeyboardInterrupt:
                self.logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(300)


app = typer.Typer(help="Kp Index Space Weather Monitor", add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    once: bool = typer.Option(False, "--once", help="Run single check and exit"),
    continuous: bool = typer.Option(False, "--continuous", help="Run continuous monitoring"),
    test: bool = typer.Option(False, "--test", help="Run test mode with sample data"),
):
    """
    Main function with command line interface.
    """
    selected = [flag for flag in (once, continuous, test) if flag]
    if len(selected) == 0:
        raise typer.BadParameter("One of --once, --continuous, or --test must be specified")
    if len(selected) > 1:
        raise typer.BadParameter(
            "Options --once, --continuous, and --test are mutually exclusive i.e., only one can be selected."
        )

    config = MonitorConfig.from_yaml()
    log_suffix = "once" if once else "continuous" if not test else "test"
    monitor = KpMonitor(config, log_suffix=log_suffix)

    if once:
        monitor.run_single_check(test=False)
    elif test:
        monitor.run_single_check(test=True)
    elif continuous:
        monitor.run_continuous_monitoring()


if __name__ == "__main__":
    app()
