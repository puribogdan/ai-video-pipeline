def _cleanup_all_unfinished(self):
    """Clean up ONLY incomplete/failed productions (not status 3 = Done)"""
    try:
        url = f"{self.base_url}/productions.json"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            return
        
        productions = response.json().get("data", [])
        deleted = 0
        
        for prod in productions:
            status = prod.get("status")
            uuid = prod.get("uuid")
            
            # Only delete incomplete/failed jobs
            # Status 3 = Done (keep these!)
            # Status 2 = Error (delete)
            # Status 9 = Incomplete (delete)
            # Status 0,1,4,5,6,7,8,10,11,12,13,14,15 = Processing/waiting (delete if old)
            
            should_delete = False
            
            # Always delete errors and incomplete
            if status in [2, 9]:
                should_delete = True
                logger.info(f"Deleting failed/incomplete: {uuid} (status: {status})")
            
            # Delete old processing jobs (stuck for >1 hour)
            elif status in [0, 1, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]:
                created_time = prod.get("creation_time")
                if created_time:
                    from datetime import datetime, timedelta
                    try:
                        created_dt = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                        age = datetime.now(created_dt.tzinfo) - created_dt
                        
                        if age > timedelta(hours=1):
                            should_delete = True
                            logger.info(f"Deleting old stuck job: {uuid} (status: {status}, age: {age})")
                    except:
                        pass
            
            # DON'T delete status 3 (Done) - these are successful!
            
            if should_delete and uuid:
                delete_url = f"{self.base_url}/production/{uuid}.json"
                del_response = requests.delete(delete_url, headers=headers, timeout=30)
                if del_response.status_code in [200, 204]:
                    deleted += 1
        
        logger.info(f"Cleanup complete: {deleted} unfinished productions deleted")
        
    except Exception as e:
        logger.warning(f"Bulk cleanup error: {e}")