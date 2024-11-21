from datetime import datetime, timedelta
from geopy.distance import geodesic
from concurrent.futures import ThreadPoolExecutor

def handle_unvisitable(locations, clusters, daily_start, daily_end):
    """
    Fitting unvisitable locations data into other clusters based on their proximity and time constraints (if possible)

    Parameters:
        locations (list): Unvisitable locations list
        clusters (dict): Dictionary containing existing clusters and their schedules
        daily_start (str): Daily starting time in "HH:MM" format
        daily_end (str): Daily ending time in "HH:MM" format

    Returns:
        dict: A dictionary with updated clusters and remaining unvisitable locations
    """
    
    new_unvisitable = []
    for location in locations:
        fit = False
        for cluster_id, cluster_schedule in clusters.items():
            result = schedule_single_location(location, cluster_schedule["schedule"], daily_start, daily_end)
            if result:
                cluster_schedule["schedule"].append(result)
                cluster_schedule["schedule"].sort(key=lambda x: x["start_time"])
                fit = True
                break
        if not fit:
            new_unvisitable.append(location)

    return {"clusters": clusters, "unvisitable": new_unvisitable}

def schedule_single_location(location, current_schedule, daily_start, daily_end):
    """
    Schedule a single location within a cluster, ensuring it fits within the daily and business hour constraints.

    Parameters:
        location (dict): The location to schedule, with details such as opening hours and duration.
        current_schedule (list): Current schedule of the cluster.
        daily_start (str): The daily starting time in "HH:MM" format.
        daily_end (str): The daily ending time in "HH:MM" format.

    Returns:
        dict or None: The scheduled entry or None if the location cannot be scheduled.
    """
    
    # Convert string to datetime object
    daily_start_time = datetime.strptime(daily_start, "%H:%M")
    daily_end_time = datetime.strptime(daily_end, "%H:%M")
    opening_time = datetime.strptime(location.opening_hours, "%H:%M")
    closing_time = datetime.strptime(location.closing_hours, "%H:%M")

    # Determine the earliest possible start time
    start_time = max(daily_start_time, opening_time)
    end_time = start_time + timedelta(hours=location.duration)

    # Ensure the location fits within business hours and daily time constraints
    if end_time > min(daily_end_time, closing_time):
        return None

    # Check availability in the schedule
    # Iterating through the current schedule
    for i, event in enumerate(current_schedule):
        
        # Convert all event start and end times to datetime objects
        event_start = datetime.strptime(event["start_time"], "%H:%M")
        event_end = datetime.strptime(event["end_time"], "%H:%M")

        # If location fits before the first event
        if i == 0 and end_time <= event_start:
            return create_schedule_entry(location, start_time, end_time)

        # If location fits between events
        if i > 0:
            prev_event_end = datetime.strptime(current_schedule[i - 1]["end_time"], "%H:%M")
            if prev_event_end + timedelta(hours=location.duration) <= event_start:
                start_time = prev_event_end
                end_time = start_time + timedelta(hours=location.duration)
                return create_schedule_entry(location, start_time, end_time)

    # If location fits after the last event
    if current_schedule:
        last_event_end = datetime.strptime(current_schedule[-1]["end_time"], "%H:%M")
        if last_event_end + timedelta(hours=location.duration) <= min(daily_end_time, closing_time):
            start_time = last_event_end
            end_time = start_time + timedelta(hours=location.duration)
            return create_schedule_entry(location, start_time, end_time)

    # If no events exist and location fits in the day
    if not current_schedule and end_time <= min(daily_end_time, closing_time):
        return create_schedule_entry(location, start_time, end_time)

    return None  # Cannot fit

def create_schedule_entry(location, start_time, end_time):
    """
    Returning schedule entry with the location's details and scheduled times.
    """
    return {
        "name": location.name,
        "coordinates": location.coordinates,
        "start_time": start_time.strftime("%H:%M"),
        "end_time": end_time.strftime("%H:%M")
    }

def schedule_cluster_with_priorities(cluster, daily_start="08:00", daily_end="20:00"):
    """
    Schedule locations within a cluster, balancing proximity and business hours.

    Parameters:
        cluster (list): List of locations in the cluster.
        daily_start (str): The daily starting time in "HH:MM" format.
        daily_end (str): The daily ending time in "HH:MM" format.

    Return a dict of scheduled locations and any unvisitable locations.
    """
    schedule = []
    unvisitable = []
    current_time = datetime.strptime(daily_start, "%H:%M")

    while cluster:
        if not schedule:
            # Select the location with the earliest opening time
            next_location = min(cluster, key=lambda x: datetime.strptime(x.opening_hours, "%H:%M"))
            cluster.remove(next_location)
            reason = "Chosen as the first location based on earliest opening hours"
        else:
            # Calculate scores for remaining locations based on proximity and time flexibility
            last_location = schedule[-1]
            next_location = min(cluster, key=lambda loc: calculate_score(loc, last_location, current_time))
            cluster.remove(next_location)
            distance_to_last = geodesic(last_location["coordinates"], next_location.coordinates).kilometers
            reason = f"Chosen based on proximity ({distance_to_last:.2f} km) and business hours compatibility"

        # Schedule the selected location
        result = schedule_single_location(next_location, schedule, daily_start, daily_end)
        if result:
            result["reason"] = reason
            schedule.append(result)
            current_time = datetime.strptime(result["end_time"], "%H:%M")
        else:
            unvisitable.append(next_location)

    # Add proximity details to each scheduled location
    for i in range(len(schedule) - 1):
        distance_to_next = geodesic(schedule[i]["coordinates"], schedule[i + 1]["coordinates"]).kilometers
        schedule[i]["proximity_to_next"] = f"{distance_to_next:.2f} km"

    if schedule:
        schedule[-1]["proximity_to_next"] = "N/A"  # Last location has no next location

    return {"schedule": schedule, "unvisitable": unvisitable}

def calculate_score(location, last_location, current_time):
    """
    Calculate a weighted score for scheduling a location.
    """
    proximity_score = geodesic(last_location["coordinates"], location.coordinates).kilometers
    closing_time = datetime.strptime(location.closing_hours, "%H:%M")
    time_flexibility = max((closing_time - current_time).total_seconds() / 3600, 0.1)
    return proximity_score + (1 / time_flexibility) # Weighted score (lower is better).

def parallel_schedule_clusters(clusters, daily_start="08:00", daily_end="20:00", num_threads=4):
    """
    Schedulling multiple clusters in parallel using multithreading.
    
    Return a dict of scheduled clusters and unvisitable locations.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            cluster_id: executor.submit(iterative_schedule_cluster, cluster_locations, daily_start, daily_end)
            for cluster_id, cluster_locations in clusters.items()
        }
        for cluster_id, future in futures.items():
            results[cluster_id] = future.result()

    return results

# 20 Iterations overkill?
def iterative_schedule_cluster(cluster, daily_start="08:00", daily_end="20:00", max_iterations=20):
    """
    Find the best schedule for a single cluster from multiple iterations
    
    Return a dict of best schedule and any remaining unvisitable locations
    """
    best_schedule = None
    best_unvisitable = None
    best_score = float("inf")

    for _ in range(max_iterations):
        result = schedule_cluster_with_priorities(cluster[:], daily_start, daily_end)
        unvisitable_count = len(result["unvisitable"])
        schedule_length = len(result["schedule"])
        score = unvisitable_count * 1000 - schedule_length

        if score < best_score:
            best_score = score
            best_schedule = result["schedule"]
            best_unvisitable = result["unvisitable"]

        # Stop early if no unvisitable locations
        if unvisitable_count == 0:
            break

    return {"schedule": best_schedule, "unvisitable": best_unvisitable}