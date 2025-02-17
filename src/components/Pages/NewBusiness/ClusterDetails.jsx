import React from 'react';
import { useLocation } from 'react-router-dom';
import './ClusterDetails.css'; // Import the CSS file for styling

const ClusterDetails = () => {
  const location = useLocation();
  const clusterInfo = location.state?.clusterInfo;

  // Extract the first neighborhood for mapping
  const firstNeighborhood = clusterInfo?.neighborhood_areas[0] || '';

  return (
    <div className="details-container">
      <div className="details-left">
        <h2>Cluster {clusterInfo?.cluster}</h2>
        <h4>Recommended Neighborhoods:</h4>
        <ul>
          {clusterInfo?.neighborhood_areas.map((neighborhood, index) => (
            <li key={index}>{neighborhood}</li>
          ))}
        </ul>
      </div>
      <div className="details-right">
        {firstNeighborhood ? (
          <iframe
            title="Google Maps"
            src={`https://www.google.com/maps/embed/v1/place?key=YOUR_GOOGLE_MAPS_API_KEY&q=${encodeURIComponent(
              firstNeighborhood
            )}`}
            width="100%"
            height="100%"
            style={{ border: 0 }}
            allowFullScreen
          ></iframe>
        ) : (
          <p>Map not available</p>
        )}
      </div>
    </div>
  );
};

export default ClusterDetails;
