import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx

df = pd.read_csv('BA.csv')
# BA is a dataset containing the location, date, and age of fatal bear attack victims. It has categorical and numerical data with missing age,
# latitude, and longitude, and gender rows. The missing ages are replaced with the mean of the column and the missing genders are assumed to be the 
# mode of the column. Missing Latitude and Longitude rows were deleted as they couldn't be verified. The following code is to determine central
# points of bear attacks based on past records.

df = df.dropna(subset=['Latitude', 'Longitude'])
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
# Database processing

numerical_features = ['Longitude', 'Latitude']
X = df[numerical_features]

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 6
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

df['Cluster'] = kmeans.labels_

centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_longitude = centroids[:, 0]
centroid_latitude = centroids[:, 1]

# Create a GeoDataFrame for bear attack points and centroids
geometry_points = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf_points = gpd.GeoDataFrame(df, geometry=geometry_points, crs="EPSG:4326")

centroids_df = pd.DataFrame({'Longitude': centroid_longitude, 'Latitude': centroid_latitude})
geometry_centroids = [Point(xy) for xy in zip(centroids_df['Longitude'], centroids_df['Latitude'])]
gdf_centroids = gpd.GeoDataFrame(centroids_df, geometry=geometry_centroids, crs="EPSG:4326")

# Plotting the data on a map with OpenStreetMap and focusing on North America
fig, ax = plt.subplots(figsize=(12, 8))

# Plot bear attack points
gdf_points.plot(ax=ax, markersize=10, alpha=0.7, column='Cluster', cmap='viridis', legend=True)

# Plot centroids
gdf_centroids.plot(ax=ax, color='red', marker='X', markersize=100, label='Centroids')

# Add OpenStreetMap basemap (zoomed on North America)
ax.set_xlim([-180, -50])  # Longitude range for North America
ax.set_ylim([10, 80])     # Latitude range for North America
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=4, crs=gdf_points.crs.to_string())

# Set plot titles and labels
ax.set_title('Bear Attack Locations with Clusters Overlayed on North America Map', fontsize=15)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
plt.legend()
plt.show()

