import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KMAClimateAnalyzer:
    def __init__(self):
        # Extended station to region code mapping
        self.station_to_region = {
    # Seoul Metropolitan City (서울특별시)
    '108': '11',  # Seoul
    
    # Busan Metropolitan City (부산광역시)
    '159': '26',  # Busan
    
    # Daegu Metropolitan City (대구광역시)
    '143': '27',  # Daegu
    
    # Incheon Metropolitan City (인천광역시)
    '112': '28',  # Incheon
    '201': '28',  # Ganghwa
    
    # Gwangju Metropolitan City (광주광역시)
    '156': '29',  # Gwangju
    
    # Daejeon Metropolitan City (대전광역시)
    '133': '30',  # Daejeon
    
    # Ulsan Metropolitan City (울산광역시)
    '152': '31',  # Ulsan
    
    # Sejong Special Self-Governing City (세종특별자치시)
    '239': '36', # Sejong
    
    # Gyeonggi Province (경기도)
    '098': '41',  # Dongducheon (Station 1)
    '099': '41',  # Icheon
    '101': '41',  # Suwon (Station 1)
    '102': '41',  # Yangpyeong (Station 1)
    '119': '41',  # Suwon (Station 2)
    '202': '41',  # Yangpyeong (Station 2)
    '203': '41',  # Dongducheon (Station 2)
    
    # Gangwon Province (강원도)
    '090': '42',  # Sokcho (Station 1)
    '093': '42',  # Daegwallyeong (Station 1)
    '095': '42',  # Cheorwon
    '100': '42',  # Daegwallyeong (Station 2)
    '104': '42',  # North Gangneung
    '105': '42',  # Gangneung (Station 1)
    '106': '42',  # Sokcho (Station 2)
    '114': '42',  # Wonju
    '177': '42',  # Hongcheon (Station 2)
    '211': '42',  # Hongcheon
    '212': '42',  # Taebaek (Station 1)
    '216': '42',  # Gangneung (Station 2)
    '217': '42',  # Taebaeksan
    
    # Chungcheongbuk Province (충청북도)
    '127': '43',  # Chungju
    '131': '43',  # Cheongju (Station 1)
    '135': '43',  # Boeun
    '221': '43',  # Jecheon (Station 1)
    '226': '43',  # Jecheon (Station 2)
    '276': '43',  # Cheongju (Station 2)
    
    # Chungcheongnam Province (충청남도)
    '129': '44',  # Seosan (Station 1)
    '232': '44',  # Cheonan (Station 1)
    '235': '44',  # Boryeong
    '236': '44',  # Buyeo
    '238': '44',  # Geumsan
    '281': '44',  # Seosan (Station 2)
    
    # Jeollabuk Province (전라북도)
    '140': '45',  # Gunsan
    '146': '45',  # Jeonju (Station 1)
    '174': '45',  # Imsil
    '243': '45',  # Namwon (Station 1)
    '244': '45',  # Jeongeup (Station 1)
    '245': '45',  # Gochang
    '247': '45',  # Namwon (Station 2)
    '248': '45',  # Jeongeup (Station 2)
    '283': '45',  # Jeonju (Station 2)
    '284': '45',  # Geochang
    '285': '45',  # Namwon (Station 3)
    
    # Jeollanam Province (전라남도)
    '165': '46',  # Mokpo
    '168': '46',  # Yeosu (Station 1)
    '169': '46',  # Suncheon
    '170': '46',  # Wando
    '251': '46',  # Jangheung (Station 1)
    '252': '46',  # Yeonggwang
    '253': '46',  # Yeongsanpo
    '254': '46',  # Jangseong
    '255': '46',  # Gwangyang
    '260': '46',  # Jangheung (Station 2)
    '261': '46',  # Haenam
    '262': '46',  # Goheung
    '288': '46',  # Yeosu (Station 2)
    
    # Gyeongsangbuk Province (경상북도)
    '115': '47',  # Ulleungdo
    '121': '47',  # Yeongcheon (Station 1)
    '130': '47',  # Pohang (Station 1)
    '136': '47',  # Andong
    '137': '47',  # Uljin
    '138': '47',  # Pohang (Station 2)
    '172': '47',  # Yeongcheon (Station 2)
    '257': '47',  # Yeongdeok (Station 1)
    '258': '47',  # Uiseong (Station 1)
    '259': '47',  # Gumi
    '263': '47',  # Mungyeong (Station 1)
    '271': '47',  # Bonghwa
    '272': '47',  # Yeongdeok (Station 2)
    '273': '47',  # Uiseong (Station 2)
    '277': '47',  # Youngju
    '278': '47',  # Mungyeong (Station 2)
    '279': '47',  # Yeongcheon (Station 3)
    '289': '47',  # Pohang (Station 3)
    
    # Gyeongsangnam Province (경상남도)
    '155': '48',  # Changwon
    '162': '48',  # Tongyeong
    '192': '48',  # Geochang (Station 1)
    '264': '48',  # Sancheong
    '266': '48',  # Geochang (Station 2)
    '268': '48',  # Hapcheon (Station 1)
    '294': '48',  # Namhae
    '295': '48',  # Hapcheon (Station 2)
    
    # Jeju Province (제주도)
    '184': '49',  # Jeju
    '185': '49',  # Gosan
    '188': '49',  # Seongsan
    '189': '49',  # Seogwipo
        }
        
        self.region_names = {
            '11': 'Seoul',
            '26': 'Busan',
            '27': 'Daegu',
            '28': 'Incheon',
            '29': 'Gwangju',
            '30': 'Daejeon',
            '31': 'Ulsan',
            '36': 'Sejong',
            '41': 'Gyeonggi',
            '42': 'Gangwon',
            '43': 'Chungbuk',
            '44': 'Chungnam',
            '45': 'Jeonbuk',
            '46': 'Jeonnam',
            '47': 'Gyeongbuk',
            '48': 'Gyeongnam',
            '49': 'Jeju'
        }
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the climate data according to data type
        """
        
        # Ensure DateTime is datetime index
        if 'DateTime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
            df = df.dropna(subset=['DateTime'])
            df = df.sort_values('DateTime')
            df = df.set_index('DateTime')

        
        # 1. Time-based interpolation for continuous measurements
        continuous_cols = [
            'AvgTemp', 'MinTemp', 'MaxTemp',
            'AvgSeaLevelPressure', 'AvgGroundTemp',
            'AvgHumidity', 'CloudCover', 'MidLowCloudCover',
            'AvgSoilTemp5cm', 'AvgSoilTemp10cm', 'AvgSoilTemp20cm',
            'AvgSoilTemp30cm', 'SoilTemp0_5m', 'SoilTemp1_0m',
            'SoilTemp1_5m', 'SoilTemp3_0m', 'SoilTemp5_0m',
            'AvgVaporPressure', 'AvgDewPoint'
            ]
        
        df[continuous_cols] = df[continuous_cols].interpolate(method='time')
        
        # 2. Fill with 0 for cumulative measurements
        cumulative_cols = [
            'Rainfall', 'TotalLargeEvaporation',
            'SunshineHours', 'DailySolarRadiation', 'FogDuration',
            'Rain9to9', 'Max10minRain', 'Max1hrRain',
            'NewSnow3hr', 'TotalSmallEvaporation', 'RainfallHours'
            ]
        available_cumulative = [col for col in cumulative_cols if col in df.columns]
        df[available_cumulative] = df[available_cumulative].fillna(0)
        
        # 3. Special handling for wind direction
        def handle_wind_direction(x):
            if pd.isna(x).all():
                return None
            valid_directions = x.dropna()
            if valid_directions.empty:
                return None
            return valid_directions.mode().iloc[0]
        
        # Apply wind direction handling
        df['MaxWindDir'] = df.groupby('StationCode')['MaxWindDir'].transform(handle_wind_direction)
        
        # 4. Forward fill for remaining columns
        remaining_cols = [col for col in df.columns if col not in continuous_cols + cumulative_cols]
        df[remaining_cols] = df[remaining_cols].ffill().bfill()
        
        df = df.reset_index()
            
        return df
    
    def handle_missing_values_after_grouping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values at the region level after grouping.
        Applies interpolation and fills missing values based on the type of measurement.
        """
        
        if 'DateTime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
            df = df.dropna(subset=['DateTime'])
            df = df.sort_values('DateTime')
            df = df.set_index('DateTime')

        # Continuous variables: interpolate by region
        continuous_cols = [
            'AvgTemp', 'MinTemp', 'MaxTemp',
            'AvgSeaLevelPressure', 'AvgGroundTemp',
            'AvgHumidity', 'CloudCover', 'MidLowCloudCover',
            'AvgSoilTemp5cm', 'AvgSoilTemp10cm', 'AvgSoilTemp20cm',
            'AvgSoilTemp30cm', 'SoilTemp0_5m', 'SoilTemp1_0m',
            'SoilTemp1_5m', 'SoilTemp3_0m', 'SoilTemp5_0m',
            'AvgVaporPressure', 'AvgDewPoint'
        ]
        
        for col in continuous_cols:
            if col in df.columns:
                df[col] = df.groupby('RegionCode')[col].transform(lambda group: group.interpolate(method='time'))
    
        # Cumulative variables: fill missing with 0
        cumulative_cols = [
            'Rainfall', 'TotalLargeEvaporation',
            'SunshineHours', 'DailySolarRadiation', 'FogDuration',
            'Rain9to9', 'Max10minRain', 'Max1hrRain',
            'NewSnow3hr', 'TotalSmallEvaporation', 'RainfallHours'
        ]
        for col in cumulative_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
    
        # Directional data: fill using mode within each region
        directional_cols = ['MaxWindDir']
        for col in directional_cols:
            if col in df.columns:
                df[col] = df.groupby('RegionCode')[col].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    
        df = df.reset_index()
    
        return df

    def process_data(self, file_path: str) -> pd.DataFrame:
        """
        Process KMA climate data from CSV file with specific regional aggregation methods
        """
        try:
            logger.info(f"Reading data from {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Error: File not found at {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read the file and print initial information
            df = pd.read_csv(file_path, encoding='euc-kr')
            print("\nInitial data shape:", df.shape)
            print("\nInitial columns:", df.columns.tolist())
            print("\nFirst few rows of raw data:")
            print(df.head())
            
            # Rename columns
            column_mapping = {
                '지점': 'StationCode',
                '일시': 'DateTime',
                '평균기온(°C)': 'AvgTemp',
                '최저기온(°C)': 'MinTemp',
                '최고기온(°C)': 'MaxTemp',
                '일강수량(mm)': 'Rainfall',
                '강수 계속시간(hr)': 'RainfallHours',
                '평균 전운량(1/10)': 'CloudCover',
                '10분 최다 강수량(mm)': 'Max10minRain',
                '1시간 최다강수량(mm)': 'Max1hrRain',
                '9-9강수(mm)': 'Rain9to9',
                '최대 풍속(m/s)': 'MaxWindSpeed',
                '최대 풍속 풍향(16방위)': 'MaxWindDir',
                '평균 풍속(m/s)': 'AvgWindSpeed',
                '풍정합(100m)': 'TotalWindCalm',
                '최소 상대습도(%)': 'MinHumidity',
                '평균 상대습도(%)': 'AvgHumidity',
                '평균 증기압(hPa)': 'AvgVaporPressure',
                '평균 이슬점온도(°C)': 'AvgDewPoint',
                '평균 현지기압(hPa)': 'AvgLocalPressure',
                '평균 해면기압(hPa)': 'AvgSeaLevelPressure',
                '최고 해면기압(hPa)': 'MaxSeaLevelPressure',
                '최저 해면기압(hPa)': 'MinSeaLevelPressure',
                '합계 일조시간(hr)': 'SunshineHours',
                '가조시간(hr)': 'SolarRadiationHours',
                '1시간 최다일사량(MJ/m2)': 'Max1hrSolarRadiation',
                '합계 일사량(MJ/m2)': 'DailySolarRadiation',
                '일 최심적설(cm)': 'MaxSnowDepth',
                '일 최심신적설(cm)': 'MaxNewSnowDepth',
                '합계 3시간 신적설(cm)': 'NewSnow3hr',
                '평균 중하층운량(1/10)': 'MidLowCloudCover',
                '평균 지면온도(°C)': 'AvgGroundTemp',
                '최저 초상온도(°C)': 'MinGrassTemp',
                '평균 5cm 지중온도(°C)': 'AvgSoilTemp5cm',
                '평균 10cm 지중온도(°C)': 'AvgSoilTemp10cm',
                '평균 20cm 지중온도(°C)': 'AvgSoilTemp20cm',
                '평균 30cm 지중온도(°C)': 'AvgSoilTemp30cm',
                '0.5m 지중온도(°C)': 'SoilTemp0_5m',
                '1.0m 지중온도(°C)': 'SoilTemp1_0m',
                '1.5m 지중온도(°C)': 'SoilTemp1_5m',
                '3.0m 지중온도(°C)': 'SoilTemp3_0m',
                '5.0m 지중온도(°C)': 'SoilTemp5_0m',
                '합계 대형증발량(mm)': 'TotalLargeEvaporation',
                '합계 소형증발량(mm)': 'TotalSmallEvaporation',
                '안개 계속시간(hr)': 'FogDuration'
            }
            
            # Check for missing columns
            # Check for missing columns
            missing_columns = [col for col in column_mapping.keys() if col not in df.columns]
            if missing_columns:
                print(f"\nWarning: Missing expected columns: {missing_columns}")
                print("Available columns:", df.columns.tolist())
                
            df = df.rename(columns=column_mapping)
            print("\nColumns after renaming:", df.columns.tolist())
            
            # Convert station codes to region codes
            if 'StationCode' in df.columns:
                # Convert station codes to three-digit string format
                df['StationCode'] = df['StationCode'].astype(str).str.zfill(3)
                
                unique_stations = df['StationCode'].unique()
                print("\nUnique station codes found:", unique_stations)
                print("\nNumber of unique stations:", len(unique_stations))
                
                df['RegionCode'] = df['StationCode'].map(self.station_to_region)
                df['RegionName'] = df['RegionCode'].map(self.region_names)
                
                # Check mapped stations
                mapped_stations = df[df['RegionCode'].notna()]['StationCode'].unique()
                unmapped_stations = df[df['RegionCode'].isna()]['StationCode'].unique()
                print("\nSuccessfully mapped stations:", mapped_stations)
                print("Unmapped stations:", unmapped_stations)
                
                # Print the number of rows before and after removing unmapped stations
                print(f"\nRows before removing unmapped stations: {len(df)}")
                df = df.dropna(subset=['RegionCode'])
                print(f"Rows after removing unmapped stations: {len(df)}")
            else:
                print("Error: StationCode column not found")
                raise KeyError("StationCode column not found in the data")
                
                # Data cleaning and type conversion
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            
            # Handle missing values using the new method
            df = self.handle_missing_values(df)
            
            # Add derived features
            df['Month'] = df['DateTime'].dt.month
            df['Season'] = df['Month'].map(self._get_season)
            df['Year'] = df['DateTime'].dt.year
            
            # Define aggregation dictionary
            agg_dict = {
                'AvgTemp': 'mean',
                'MinTemp': 'min',
                'MaxTemp': 'max',
                'Rainfall': 'sum',
                'CloudCover': 'mean',
                # New aggregations
                'Max10minRain': 'max',
                'Max1hrRain': 'max',
                'RainfallHours':'mean',
                'Rain9to9': 'sum',
                'MaxWindSpeed': 'max',
                'MaxWindDir': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
                'AvgWindSpeed': 'mean',
                'TotalWindCalm': 'mean',
                'MinHumidity': 'min',
                'AvgHumidity': 'mean',
                'AvgVaporPressure': 'mean',
                'AvgDewPoint': 'mean',
                'AvgLocalPressure': 'min',
                'AvgSeaLevelPressure': 'mean',
                'MaxSeaLevelPressure': 'max',
                'MinSeaLevelPressure': 'min',
                'SunshineHours': 'mean',
                'SolarRadiationHours': 'sum',
                'Max1hrSolarRadiation': 'max',
                'DailySolarRadiation': 'sum',
                'MaxSnowDepth': 'max',
                'MaxNewSnowDepth': 'max',
                'NewSnow3hr': 'sum',
                'MidLowCloudCover': 'mean',
                'AvgGroundTemp': 'mean',
                'MinGrassTemp': 'min',
                'AvgSoilTemp5cm': 'mean',
                'AvgSoilTemp10cm': 'mean',
                'AvgSoilTemp20cm': 'mean',
                'AvgSoilTemp30cm': 'mean',
                'SoilTemp0_5m': 'mean',
                'SoilTemp1_0m': 'mean',
                'SoilTemp1_5m': 'mean',
                'SoilTemp3_0m': 'mean',
                'SoilTemp5_0m': 'mean',
                'TotalLargeEvaporation': 'sum',
                'TotalSmallEvaporation': 'sum',
                'FogDuration': 'mean'
                }
            
            # Aggregate data by region and date
            grouped = df.groupby(['RegionCode', 'RegionName', 'DateTime', 'Year', 'Month', 'Season']).agg(agg_dict).reset_index()
            
            # Handle missing values after grouping (region-level)
            grouped = self.handle_missing_values_after_grouping(grouped)

            print("\nFinal data shape:", grouped.shape)
            print("\nFinal columns:", grouped.columns.tolist())
                        
            logger.info("Data processing and regional aggregation completed successfully")
            return grouped
                    
        except Exception as e:
                logger.error(f"Error processing data: {str(e)}")
                print(f"Detailed error: {str(e)}")
                raise

    def analyze_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Perform comprehensive analysis of climate data with specific aggregation methods
        """
        analysis_results = {}
        
        # Regional statistics
        analysis_results['regional_stats'] = df.groupby('RegionName').agg({
            'AvgTemp': 'mean',
            'MinTemp': ['min', 'max'],
            'MaxTemp': ['min', 'max'],
            'Rainfall': 'sum',
            'CloudCover': 'mean'
        }).round(2)
        
        # Seasonal analysis
        analysis_results['seasonal_stats'] = df.groupby(['RegionName', 'Season']).agg({
            'AvgTemp': 'mean',
            'MinTemp': 'min',
            'MaxTemp': 'max',
            'Rainfall': 'sum',
            'CloudCover': 'mean'
        }).round(2)
        
        # Monthly trends
        analysis_results['monthly_trends'] = df.groupby(['RegionName', 'Month']).agg({
            'AvgTemp': 'mean',
            'MinTemp': 'min',
            'MaxTemp': 'max',
            'Rainfall': 'sum',
            'CloudCover': 'mean'
        }).round(2)
        
        return analysis_results

    def visualize_data(self, df: pd.DataFrame, output_dir: str = 'climate_plots'):
        """
        Create visualizations of the aggregated climate data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Average Temperature Distribution by Region
        plt.figure(figsize=(15, 8))
        sns.boxplot(x='RegionName', y='AvgTemp', data=df)
        plt.xticks(rotation=45)
        plt.title('Average Temperature Distribution by Region')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/avg_temp_distribution.png')
        plt.close()
        
        # 2. Total Rainfall by Region and Month
        plt.figure(figsize=(15, 8))
        monthly_rainfall = df.groupby(['RegionName', 'Month'])['Rainfall'].sum().unstack()
        sns.heatmap(monthly_rainfall, cmap='YlGnBu', annot=True, fmt='.0f')
        plt.title('Monthly Total Rainfall by Region')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/rainfall_patterns.png')
        plt.close()
        
        # 3. Temperature Extremes by Region
        plt.figure(figsize=(15, 8))
        for region in df['RegionName'].unique():
            region_data = df[df['RegionName'] == region]
            plt.plot(region_data['Month'].unique(), 
                    region_data.groupby('Month')['MaxTemp'].max(), 
                    label=f'{region} (Max)', linestyle='--')
            plt.plot(region_data['Month'].unique(), 
                    region_data.groupby('Month')['MinTemp'].min(), 
                    label=f'{region} (Min)', linestyle=':')
        plt.xlabel('Month')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature Extremes by Region')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/temp_extremes.png')
        plt.close()

    @staticmethod
    def _get_season(month: int) -> str:
        """Helper method to determine season from month"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    def export_results(self, df: pd.DataFrame, analysis_results: Dict[str, pd.DataFrame], 
                      output_dir: str = 'climate_results'):
        """
        Export processed data and analysis results
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Export processed data
            df.to_csv(f'{output_dir}/2020_processed_climate_data.csv', encoding='utf-8', index=False)
            
            # Export analysis results
            for name, result in analysis_results.items():
                result.to_csv(f'{output_dir}/{name}.csv', encoding='utf-8')
            
            logger.info(f"Results exported successfully to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            raise

def main():
    # Set the file path
    file_path = os.path.join(os.path.expanduser("~"), "Desktop", "Raw Data", "Climate Data 2", "2020.csv")
    
    # Initialize the analyzer
    analyzer = KMAClimateAnalyzer()
    
    try:
        # Process the data
        print("Processing data...")
        df = analyzer.process_data(file_path)
        
        # Set display options for better visualization
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', 10)
        
        # Print the first few rows of the processed data
        print("\nFirst few rows of the processed data:")
        print(df.head())
        
        # Print basic information about the dataset
        print("\nDataset Information:")
        print(f"Total number of records: {len(df)}")
        print(f"Date range: from {df['DateTime'].min()} to {df['DateTime'].max()}")
        
        # Perform analysis
        print("\nPerforming analysis...")
        analysis_results = analyzer.analyze_data(df)
        
        # Create visualizations
        print("Creating visualizations...")
        analyzer.visualize_data(df)
        
        # Export results
        print("Exporting results...")
        analyzer.export_results(df, analysis_results)
        
        # Print summary statistics
        print("\nSummary Statistics by Region:")
        print("\nAverage Temperature (°C):")
        print(analysis_results['regional_stats']['AvgTemp'])
        
        print("\nTotal Rainfall (mm):")
        print(analysis_results['regional_stats']['Rainfall'])
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()