

import asyncio
import json
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import aiohttp
import click
from google.genai import types


class ApiClient:
    """Client for interacting with the FastAPI server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to the server."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.base_url}{endpoint}"
        async with self.session.request(method, url, **kwargs) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise click.ClickException(f"HTTP {response.status}: {error_text}")
            
            if response.content_type == 'application/json':
                return await response.json()
            else:
                return {"text": await response.text()}
    
    # App management
    async def list_apps(self) -> List[str]:
        """List all available apps."""
        return await self._request("GET", "/list-apps")
    
    # Session management
    async def create_session(self, app_name: str, user_id: str, 
                           session_id: Optional[str] = None,
                           state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new session."""
        if session_id:
            endpoint = f"/apps/{app_name}/users/{user_id}/sessions/{session_id}"
        else:
            endpoint = f"/apps/{app_name}/users/{user_id}/sessions"
        
        json_data = {"state": state} if state else {}
        return await self._request("POST", endpoint, json=json_data)
    
    async def get_session(self, app_name: str, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get session details."""
        endpoint = f"/apps/{app_name}/users/{user_id}/sessions/{session_id}"
        return await self._request("GET", endpoint)
    
    async def list_sessions(self, app_name: str, user_id: str) -> List[Dict[str, Any]]:
        """List all sessions for a user."""
        endpoint = f"/apps/{app_name}/users/{user_id}/sessions"
        return await self._request("GET", endpoint)
    
    async def delete_session(self, app_name: str, user_id: str, session_id: str):
        """Delete a session."""
        endpoint = f"/apps/{app_name}/users/{user_id}/sessions/{session_id}"
        await self._request("DELETE", endpoint)
    
    # Agent interaction
    async def run_agent(self, app_name: str, user_id: str, session_id: str,
                       message: str, streaming: bool = False) -> List[Dict[str, Any]]:
        """Run the agent with a message."""
        content = types.Content(
            role="user",
            parts=[types.Part(text=message)]
        )
        
        request_data = {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
            "new_message": content.model_dump(by_alias=True),
            "streaming": streaming
        }
        
        return await self._request("POST", "/run", json=request_data)
    
    async def run_agent_streaming(self, app_name: str, user_id: str, session_id: str,
                                 message: str) -> str:
        """Run the agent with streaming response."""
        content = types.Content(
            role="user",
            parts=[types.Part(text=message)]
        )
        
        request_data = {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
            "new_message": content.model_dump(by_alias=True),
            "streaming": True
        }
        
        url = f"{self.base_url}/run_sse"
        async with self.session.post(url, json=request_data) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise click.ClickException(f"HTTP {response.status}: {error_text}")
            
            result = ""
            async for line in response.content:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith('data: '):
                    event_data = line_str[6:]  # Remove 'data: ' prefix
                    if event_data:
                        try:
                            event = json.loads(event_data)
                            if event.get('content') and event['content'].get('parts'):
                                for part in event['content']['parts']:
                                    if part.get('text'):
                                        result += part['text']
                                        print(part['text'], end='', flush=True)
                        except json.JSONDecodeError:
                            continue
            return result
    
    # Artifact management
    async def list_artifacts(self, app_name: str, user_id: str, session_id: str) -> List[str]:
        """List artifacts for a session."""
        endpoint = f"/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts"
        return await self._request("GET", endpoint)
    
    async def get_artifact(self, app_name: str, user_id: str, session_id: str,
                          artifact_name: str, version: Optional[int] = None) -> Dict[str, Any]:
        """Get an artifact."""
        if version is not None:
            endpoint = f"/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions/{version}"
        else:
            endpoint = f"/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}"
        return await self._request("GET", endpoint)
    
    async def upload_artifact(self, app_name: str, user_id: str, session_id: str,
                             file_path: str, artifact_name: Optional[str] = None) -> Dict[str, Any]:
        """Upload an artifact."""
        import os
        endpoint = f"/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/upload"
        
        filename = artifact_name or os.path.basename(file_path)
        
        with open(file_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=filename)
            if artifact_name:
                data.add_field('filename', artifact_name)
            
            return await self._request("POST", endpoint, data=data)
    
    async def delete_artifact(self, app_name: str, user_id: str, session_id: str,
                             artifact_name: str):
        """Delete an artifact."""
        endpoint = f"/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}"
        await self._request("DELETE", endpoint)
    
    # Evaluation management
    async def list_eval_sets(self, app_name: str) -> List[str]:
        """List evaluation sets."""
        endpoint = f"/apps/{app_name}/eval_sets"
        return await self._request("GET", endpoint)
    
    async def create_eval_set(self, app_name: str, eval_set_id: str):
        """Create an evaluation set."""
        endpoint = f"/apps/{app_name}/eval_sets/{eval_set_id}"
        await self._request("POST", endpoint)
    
    async def list_evals_in_set(self, app_name: str, eval_set_id: str) -> List[str]:
        """List evaluations in a set."""
        endpoint = f"/apps/{app_name}/eval_sets/{eval_set_id}/evals"
        return await self._request("GET", endpoint)


@click.group()
@click.option('--base-url', default='http://localhost:8000', 
              help='Base URL of the FastAPI server')
@click.pass_context
def cli(ctx, base_url: str):
    """Command line client for interacting with the ADK Agent FastAPI server."""
    ctx.ensure_object(dict)
    ctx.obj['base_url'] = base_url


@cli.group()
def app():
    """App management commands."""
    pass


@app.command('list')
@click.pass_context
def list_apps(ctx):
    """List all available apps."""
    async def _list_apps():
        async with ApiClient(ctx.obj['base_url']) as client:
            apps = await client.list_apps()
            if apps:
                click.echo("Available apps:")
                for app_name in apps:
                    click.echo(f"  - {app_name}")
            else:
                click.echo("No apps found.")
    
    asyncio.run(_list_apps())


@cli.group()
def session():
    """Session management commands."""
    pass


@session.command('create')
@click.argument('app_name')
@click.argument('user_id')
@click.option('--session-id', help='Specific session ID to create')
@click.option('--state-file', type=click.Path(exists=True), 
              help='JSON file containing initial state')
@click.pass_context
def create_session(ctx, app_name: str, user_id: str, session_id: Optional[str],
                  state_file: Optional[str]):
    """Create a new session."""
    async def _create_session():
        state = None
        if state_file:
            with open(state_file, 'r') as f:
                state = json.load(f)
        
        async with ApiClient(ctx.obj['base_url']) as client:
            session_data = await client.create_session(app_name, user_id, session_id, state)
            click.echo(f"Created session: {session_data['id']}")
            click.echo(json.dumps(session_data, indent=2))
    
    asyncio.run(_create_session())


@session.command('get')
@click.argument('app_name')
@click.argument('user_id')
@click.argument('session_id')
@click.pass_context
def get_session(ctx, app_name: str, user_id: str, session_id: str):
    """Get session details."""
    async def _get_session():
        async with ApiClient(ctx.obj['base_url']) as client:
            session_data = await client.get_session(app_name, user_id, session_id)
            click.echo(json.dumps(session_data, indent=2))
    
    asyncio.run(_get_session())


@session.command('list')
@click.argument('app_name')
@click.argument('user_id')
@click.pass_context
def list_sessions(ctx, app_name: str, user_id: str):
    """List all sessions for a user."""
    async def _list_sessions():
        async with ApiClient(ctx.obj['base_url']) as client:
            sessions = await client.list_sessions(app_name, user_id)
            if sessions:
                click.echo(f"Sessions for {user_id} in {app_name}:")
                for session_data in sessions:
                    click.echo(f"  - {session_data['id']} (created: {session_data.get('created_at', 'unknown')})")
            else:
                click.echo("No sessions found.")
    
    asyncio.run(_list_sessions())


@session.command('delete')
@click.argument('app_name')
@click.argument('user_id')
@click.argument('session_id')
@click.pass_context
def delete_session(ctx, app_name: str, user_id: str, session_id: str):
    """Delete a session."""
    async def _delete_session():
        async with ApiClient(ctx.obj['base_url']) as client:
            await client.delete_session(app_name, user_id, session_id)
            click.echo(f"Deleted session: {session_id}")
    
    asyncio.run(_delete_session())


@cli.group()
def agent():
    """Agent interaction commands."""
    pass


@agent.command('run')
@click.argument('app_name')
@click.argument('user_id')
@click.argument('session_id')
@click.argument('message')
@click.option('--streaming', is_flag=True, help='Use streaming response')
@click.pass_context
def run_agent(ctx, app_name: str, user_id: str, session_id: str, message: str, streaming: bool):
    """Run the agent with a message."""
    async def _run_agent():
        async with ApiClient(ctx.obj['base_url']) as client:
            if streaming:
                click.echo(f"[{user_id}]: {message}")
                click.echo(f"[agent]: ", end='')
                await client.run_agent_streaming(app_name, user_id, session_id, message)
                click.echo()  # New line after streaming
            else:
                events = await client.run_agent(app_name, user_id, session_id, message, streaming)
                click.echo(f"[{user_id}]: {message}")
                
                for event in events:
                    if event.get('content') and event['content'].get('parts'):
                        text_parts = []
                        for part in event['content']['parts']:
                            if part.get('text'):
                                text_parts.append(part['text'])
                        if text_parts:
                            author = event.get('author', 'agent')
                            click.echo(f"[{author}]: {''.join(text_parts)}")
    
    asyncio.run(_run_agent())


@agent.command('chat')
@click.argument('app_name')
@click.argument('user_id')
@click.argument('session_id')
@click.option('--streaming', is_flag=True, help='Use streaming response')
@click.pass_context
def chat_with_agent(ctx, app_name: str, user_id: str, session_id: str, streaming: bool):
    """Start an interactive chat with the agent."""
    async def _chat():
        async with ApiClient(ctx.obj['base_url']) as client:
            click.echo(f"Starting chat with {app_name} (session: {session_id})")
            click.echo("Type 'exit' to quit")
            click.echo()
            
            while True:
                try:
                    message = input(f"[{user_id}]: ").strip()
                    if not message:
                        continue
                    if message.lower() == 'exit':
                        break
                    
                    if streaming:
                        click.echo(f"[agent]: ", end='')
                        await client.run_agent_streaming(app_name, user_id, session_id, message)
                        click.echo()  # New line after streaming
                    else:
                        events = await client.run_agent(app_name, user_id, session_id, message, streaming)
                        
                        for event in events:
                            if event.get('content') and event['content'].get('parts'):
                                text_parts = []
                                for part in event['content']['parts']:
                                    if part.get('text'):
                                        text_parts.append(part['text'])
                                if text_parts:
                                    author = event.get('author', 'agent')
                                    click.echo(f"[{author}]: {''.join(text_parts)}")
                    
                    click.echo()  # Extra line for readability
                    
                except KeyboardInterrupt:
                    click.echo("\nExiting chat...")
                    break
                except Exception as e:
                    click.echo(f"Error: {e}")
    
    asyncio.run(_chat())


@cli.group()
def artifact():
    """Artifact management commands."""
    pass


@artifact.command('list')
@click.argument('app_name')
@click.argument('user_id')
@click.argument('session_id')
@click.pass_context
def list_artifacts(ctx, app_name: str, user_id: str, session_id: str):
    """List artifacts for a session."""
    async def _list_artifacts():
        async with ApiClient(ctx.obj['base_url']) as client:
            artifacts = await client.list_artifacts(app_name, user_id, session_id)
            if artifacts:
                click.echo(f"Artifacts in session {session_id}:")
                for artifact_name in artifacts:
                    click.echo(f"  - {artifact_name}")
            else:
                click.echo("No artifacts found.")
    
    asyncio.run(_list_artifacts())


@artifact.command('get')
@click.argument('app_name')
@click.argument('user_id')
@click.argument('session_id')
@click.argument('artifact_name')
@click.option('--version', type=int, help='Specific version to retrieve')
@click.option('--output', type=click.Path(), help='Output file path')
@click.pass_context
def get_artifact(ctx, app_name: str, user_id: str, session_id: str,
                artifact_name: str, version: Optional[int], output: Optional[str]):
    """Get an artifact."""
    async def _get_artifact():
        async with ApiClient(ctx.obj['base_url']) as client:
            artifact = await client.get_artifact(app_name, user_id, session_id, artifact_name, version)
            
            if output:
                with open(output, 'w') as f:
                    json.dump(artifact, f, indent=2)
                click.echo(f"Artifact saved to: {output}")
            else:
                click.echo(json.dumps(artifact, indent=2))
    
    asyncio.run(_get_artifact())


@artifact.command('upload')
@click.argument('app_name')
@click.argument('user_id')
@click.argument('session_id')
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--name', help='Custom artifact name')
@click.pass_context
def upload_artifact(ctx, app_name: str, user_id: str, session_id: str,
                   file_path: str, name: Optional[str]):
    """Upload an artifact."""
    async def _upload_artifact():
        async with ApiClient(ctx.obj['base_url']) as client:
            result = await client.upload_artifact(app_name, user_id, session_id, file_path, name)
            click.echo(f"Uploaded artifact: {result['filename']}")
            click.echo(f"Size: {result['size']} bytes")
    
    asyncio.run(_upload_artifact())


@artifact.command('delete')
@click.argument('app_name')
@click.argument('user_id')
@click.argument('session_id')
@click.argument('artifact_name')
@click.pass_context
def delete_artifact(ctx, app_name: str, user_id: str, session_id: str, artifact_name: str):
    """Delete an artifact."""
    async def _delete_artifact():
        async with ApiClient(ctx.obj['base_url']) as client:
            await client.delete_artifact(app_name, user_id, session_id, artifact_name)
            click.echo(f"Deleted artifact: {artifact_name}")
    
    asyncio.run(_delete_artifact())


@cli.group()
def eval():
    """Evaluation management commands."""
    pass


@eval.command('list-sets')
@click.argument('app_name')
@click.pass_context
def list_eval_sets(ctx, app_name: str):
    """List evaluation sets."""
    async def _list_eval_sets():
        async with ApiClient(ctx.obj['base_url']) as client:
            eval_sets = await client.list_eval_sets(app_name)
            if eval_sets:
                click.echo(f"Evaluation sets for {app_name}:")
                for eval_set in eval_sets:
                    click.echo(f"  - {eval_set}")
            else:
                click.echo("No evaluation sets found.")
    
    asyncio.run(_list_eval_sets())


@eval.command('create-set')
@click.argument('app_name')
@click.argument('eval_set_id')
@click.pass_context
def create_eval_set(ctx, app_name: str, eval_set_id: str):
    """Create an evaluation set."""
    async def _create_eval_set():
        async with ApiClient(ctx.obj['base_url']) as client:
            await client.create_eval_set(app_name, eval_set_id)
            click.echo(f"Created evaluation set: {eval_set_id}")
    
    asyncio.run(_create_eval_set())


@eval.command('list-evals')
@click.argument('app_name')
@click.argument('eval_set_id')
@click.pass_context
def list_evals_in_set(ctx, app_name: str, eval_set_id: str):
    """List evaluations in a set."""
    async def _list_evals():
        async with ApiClient(ctx.obj['base_url']) as client:
            evals = await client.list_evals_in_set(app_name, eval_set_id)
            if evals:
                click.echo(f"Evaluations in set {eval_set_id}:")
                for eval_id in evals:
                    click.echo(f"  - {eval_id}")
            else:
                click.echo("No evaluations found in set.")
    
    asyncio.run(_list_evals())


if __name__ == '__main__':
    cli()