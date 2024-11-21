from pydantic import BaseModel
from typing import List, Optional

class Location(BaseModel):
    name: str
    coordinates: List[float]
    # Default to 08:00
    opening_hours: Optional[str] = "08:00" 
    # Default to 20:00
    closing_hours: Optional[str] = "20:00"
    duration: int

class ClusteringInput(BaseModel):
    points: List[Location]
    num_clusters: int
    # Default to 08:00
    daily_start_time: Optional[str] = "08:00"  
    # Default to 20:00
    daily_end_time: Optional[str] = "20:00"    