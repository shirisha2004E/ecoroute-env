from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from server.models import EcoRouteAction, EcoRouteObservation, EcoRouteState

class EcoRouteEnv(EnvClient[EcoRouteAction, EcoRouteObservation, EcoRouteState]):
    def _step_payload(self, action: EcoRouteAction) -> dict:
        return {"next_location_id": action.next_location_id}
    
    def _parse_result(self, payload: dict) -> StepResult[EcoRouteObservation]:
        obs_data = payload.get("observation", {})
        obs = EcoRouteObservation(
            current_location=obs_data.get("current_location", 0),
            packages_delivered=obs_data.get("packages_delivered", []),
            remaining_packages=obs_data.get("remaining_packages", []),
            fuel_used=obs_data.get("fuel_used", 0.0),
            time_elapsed=obs_data.get("time_elapsed", 0.0),
            total_reward_so_far=obs_data.get("total_reward_so_far", 0.0),
            weather=obs_data.get("weather", "clear"),
            traffic_level=obs_data.get("traffic_level", 0.0),
            legal_actions=obs_data.get("legal_actions", [])
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False)
        )
    
    def _parse_state(self, payload: dict) -> EcoRouteState:
        return EcoRouteState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            done=payload.get("done", False),
            score=payload.get("score", 0.0),
            task_level=payload.get("task_level", "easy"),
            metrics=payload.get("metrics", {})
        )