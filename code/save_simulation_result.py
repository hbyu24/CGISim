import os
import json
import datetime
import html
from pathlib import Path
import re
import shutil


class SimulationResultSaver:
    """
    Class for saving simulation results, including documents, meetings, and component measurements.
    Organizes results in a structured folder hierarchy with both JSON and HTML formats.
    """

    def __init__(self, base_dir="result"):
        """
        Initialize the result saver with a base directory.

        Args:
            base_dir (str): Base directory for saving results
        """
        self.base_dir = base_dir

    def set_base_dir(self, new_base_dir):
        """Update the base directory for saving results."""
        self.base_dir = new_base_dir
        return self.base_dir

    def create_simulation_folder(self, num_agents,ceo_mbti=None):
        """
        Create a timestamped folder for this simulation and its subfolders.

        Args:
            num_agents (int): Number of agents in the simulation

        Returns:
            str: Path to the created simulation folder
        """
        # Update timestamp format to include hours, minutes, seconds, and milliseconds
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y_%m_%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
        folder_name = f"{timestamp}_agents{num_agents}_{ceo_mbti}"
        sim_path = os.path.join(self.base_dir, folder_name)

        # Create necessary subfolders
        os.makedirs(os.path.join(sim_path, "documents"), exist_ok=True)
        os.makedirs(os.path.join(sim_path, "meetings"), exist_ok=True)
        os.makedirs(os.path.join(sim_path, "components"), exist_ok=True)

        # Print confirmation message with proper path
        print(f"Created simulation folder at: {os.path.abspath(sim_path)}")

        return sim_path

    def interactive_document_to_html(self, doc):
        """
        Convert an interactive document to HTML format with a left sidebar TOC.
        Only includes H1 (top-level) headings in the TOC.
        """
        if not doc:
            return "<html><body><p>Empty document</p></body></html>"

        # Get text content
        try:
            content = doc.text()
        except:
            try:
                # Fallback if doc is a Document but not InteractiveDocument
                content = "".join([str(c) for c in doc.contents()])
            except:
                content = str(doc)

        # Escape content for HTML first to avoid double escaping later
        content_escaped = html.escape(content)

        # Parse headers to generate a table of contents - ONLY LEVEL 1 HEADERS
        toc_items = []
        processed_content = content_escaped

        # Find all H1 headers (# Header format)
        h1_pattern = r'(^|\n)# (.+?)(\n|$)'
        matches = list(re.finditer(h1_pattern, content_escaped))

        # Process each H1 header
        for i, match in enumerate(matches):
            header_text = match.group(2).strip()
            toc_id = f"h1-{i}"

            # Store in TOC
            toc_items.append({
                "text": header_text,
                "id": toc_id
            })

            # Replace in content - wrap with proper HTML
            header_with_id = f'{match.group(1)}<h1 id="{toc_id}">{header_text}</h1>{match.group(3)}'
            processed_content = processed_content.replace(match.group(0), header_with_id, 1)

        # Convert other headers (h2-h6) without adding to TOC
        for level in range(2, 7):
            hx_pattern = r'(^|\n)' + ('#' * level) + r' (.+?)(\n|$)'

            def header_replacer(match):
                start = match.group(1)
                text = match.group(2)
                end = match.group(3)
                return f'{start}<h{level}>{text}</h{level}>{end}'

            processed_content = re.sub(hx_pattern, header_replacer, processed_content)

        # Generate left sidebar TOC
        toc_html = ''
        if toc_items:
            toc_html = '<div class="sidebar-toc"><h2>Table of Contents</h2><ul>'
            for item in toc_items:
                toc_html += f'<li><a href="#{item["id"]}">{item["text"]}</a></li>'
            toc_html += '</ul></div>'

        # Convert newlines to <br>
        processed_content = processed_content.replace('\n', '<br>\n')

        # Add paragraph breaks for readability
        processed_content = re.sub(r'<br>\n<br>\n', '</p><p>', processed_content)
        processed_content = f"<p>{processed_content}</p>"

        # Build HTML document with left sidebar layout
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simulation Document</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 0;
                    color: #333;
                    display: flex;
                    min-height: 100vh;
                }}
                .sidebar-toc {{
                    width: 250px;
                    background-color: #f8f8f8;
                    border-right: 1px solid #ddd;
                    position: fixed;
                    top: 0;
                    left: 0;
                    height: 100vh;
                    overflow-y: auto;
                    padding: 20px;
                    box-sizing: border-box;
                    z-index: 1000;
                }}
                .sidebar-toc h2 {{
                    margin-top: 0;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 5px;
                    font-size: 18px;
                }}
                .sidebar-toc ul {{
                    list-style-type: none;
                    padding-left: 0;
                }}
                .sidebar-toc li {{
                    margin-bottom: 8px;
                }}
                .sidebar-toc a {{
                    text-decoration: none;
                    color: #0066cc;
                    display: block;
                    padding: 4px 8px;
                    border-radius: 4px;
                    transition: background-color 0.2s;
                }}
                .sidebar-toc a:hover {{
                    background-color: #e8f4f8;
                    text-decoration: underline;
                }}
                .main-content {{
                    margin-left: 280px;
                    padding: 20px;
                    max-width: calc(100% - 280px);
                    flex-grow: 1;
                    box-sizing: border-box;
                }}
                .content {{
                    max-width: 1100px;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    color: #222;
                    margin-top: 24px;
                    margin-bottom: 12px;
                    scroll-margin-top: 24px;
                }}
                h1 {{ 
                    border-bottom: 2px solid #333; 
                    padding-bottom: 10px; 
                }}
                h2 {{ 
                    border-bottom: 1px solid #666; 
                    padding-bottom: 5px; 
                }}
                pre {{ 
                    background-color: #f4f4f4; 
                    padding: 10px; 
                    border-radius: 5px; 
                    overflow-x: auto; 
                }}
                blockquote {{ 
                    border-left: 3px solid #ccc; 
                    padding-left: 10px; 
                    color: #666; 
                }}
                p {{ 
                    margin-bottom: 1em; 
                }}
                .metadata {{
                    font-size: 0.9em;
                    color: #666;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #eee;
                }}

                /* Responsive design for smaller screens */
                @media (max-width: 768px) {{
                    body {{
                        flex-direction: column;
                    }}
                    .sidebar-toc {{
                        position: relative;
                        width: 100%;
                        height: auto;
                        border-right: none;
                        border-bottom: 1px solid #ddd;
                        overflow-y: visible;
                    }}
                    .main-content {{
                        margin-left: 0;
                        max-width: 100%;
                    }}
                }}
            </style>
        </head>
        <body>
            {toc_html}
            <div class="main-content">
                <div class="metadata">
                    Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
                <div class="content">
                    {processed_content}
                </div>
            </div>
        </body>
        </html>
        """

        return html_content

    def interactive_document_to_json(self, doc):
        """
        Convert an interactive document to JSON format.

        Args:
            doc: An interactive document object

        Returns:
            dict: JSON-serializable representation of the document
        """
        if not doc:
            return {"content": "", "contents": []}

        result = {}

        # Try to get text content
        try:
            result["content"] = doc.text()
        except:
            result["content"] = ""

        # Try to get individual contents with tags
        try:
            contents = []
            for content in doc.contents():
                contents.append({
                    "text": content.text,
                    "tags": list(content.tags) if hasattr(content, "tags") else []
                })
            result["contents"] = contents
        except:
            result["contents"] = []

        return result

    def measurements_to_html(self, measurements):
        """
        Convert measurements to HTML format.

        Args:
            measurements: A measurements object from an agent's components

        Returns:
            str: HTML representation of the measurements
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Agent Component Measurements</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    padding: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                    color: #333;
                }
                h1, h2, h3 {
                    color: #222;
                    margin-top: 20px;
                }
                h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }
                h2 { border-bottom: 1px solid #666; padding-bottom: 5px; }
                .component {
                    margin-bottom: 30px;
                    background-color: #f9f9f9;
                    padding: 15px;
                    border-radius: 5px;
                    border-left: 4px solid #0066cc;
                }
                pre {
                    background-color: #f4f4f4;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                    white-space: pre-wrap;
                }
                .measurement {
                    margin-bottom: 15px;
                    border-left: 3px solid #ccc;
                    padding-left: 10px;
                }
                .toc {
                    background-color: #f8f8f8;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 20px;
                    position: sticky;
                    top: 0;
                    z-index: 1000;
                }
                .toc h2 {
                    margin-top: 0;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 5px;
                }
                .toc ul {
                    list-style-type: none;
                    padding-left: 0;
                }
                .toc li {
                    margin-bottom: 5px;
                }
                .toc a {
                    text-decoration: none;
                    color: #0066cc;
                }
                .toc a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <h1>Agent Component Measurements</h1>
        """

        # Try to extract measurements data using Concordia's actual API
        try:
            # New approach that works with Concordia's Measurements class
            # Get channels through the _channels attribute or channels() method
            channels = {}

            if hasattr(measurements, '_channels'):
                channels = measurements._channels
            elif hasattr(measurements, 'channels') and callable(measurements.channels):
                channels = measurements.channels()
            elif hasattr(measurements, 'get_channel') and callable(measurements.get_channel):
                # If we only have get_channel method, try to get all channels that might exist
                common_channels = ['Observation', 'ObservationSummary', 'SelfPerception',
                                   'SituationPerception', 'PersonBySituation', 'ActComponent']

                for channel_name in common_channels:
                    try:
                        channel = measurements.get_channel(channel_name)
                        if channel:
                            channels[channel_name] = channel
                    except:
                        pass

            # Generate Table of Contents
            toc_items = []

            # First pass: collect component names
            component_id = 0
            for channel_name in channels.keys():
                toc_id = f"component-{component_id}"
                toc_items.append({
                    "id": toc_id,
                    "name": channel_name
                })
                component_id += 1

            # Generate TOC HTML
            toc_html = '<div class="toc"><h2>Components</h2><ul>'
            for item in toc_items:
                toc_html += f'<li><a href="#{item["id"]}">{item["name"]}</a></li>'
            toc_html += '</ul></div>'

            html_content += toc_html

            # Process each channel
            component_id = 0
            for channel_name, channel in channels.items():
                toc_id = f"component-{component_id}"
                html_content += f'<div class="component" id="{toc_id}">\n'
                html_content += f'<h2>{html.escape(channel_name)}</h2>\n'

                # Get measurements from channel
                items = []

                # Try different methods of extracting items
                if hasattr(channel, 'get_items') and callable(channel.get_items):
                    items = list(channel.get_items())
                elif hasattr(channel, 'items') and callable(channel.items):
                    items = list(channel.items())
                elif hasattr(channel, '_items'):
                    items = channel._items

                # Add measurements to HTML
                for i, measurement in enumerate(items):
                    html_content += f'<div class="measurement">\n'
                    html_content += f'<h3>Measurement {i + 1}</h3>\n'
                    html_content += f'<pre>{html.escape(str(measurement))}</pre>\n'
                    html_content += '</div>\n'

                html_content += '</div>\n'
                component_id += 1

        except Exception as e:
            html_content += f'<p>Error processing measurements: {html.escape(str(e))}</p>'
            html_content += '<p>Debug information:</p>'
            html_content += f'<pre>Measurement object type: {type(measurements)}\n'
            html_content += f'Available attributes: {dir(measurements)}</pre>'

        html_content += """
        </body>
        </html>
        """

        return html_content


    def save_meeting_results(self, simulation_folder, meeting_id, meeting_manager):

        # Clean meeting_id if needed - this ensures we use proper folder names
        cleaned_meeting_id = meeting_id.replace(" ", "_").replace("/", "_").replace("\\", "_")

        # Create meeting folder using the proper meeting_id
        meetings_path = os.path.join(simulation_folder, "meetings", cleaned_meeting_id)
        os.makedirs(meetings_path, exist_ok=True)

        try:
            # Get documents from meeting manager
            current_meeting_record = meeting_manager.current_meeting_record
            current_agent_contexts = meeting_manager.current_agent_contexts
            agent_meeting_summaries = meeting_manager.agent_meeting_summaries

            # Save current meeting record
            if current_meeting_record:
                # Save as HTML
                html_content = self.interactive_document_to_html(current_meeting_record)
                with open(os.path.join(meetings_path, "meeting_record.html"), "w", encoding="utf-8") as f:
                    f.write(html_content)

                # Save as JSON
                json_content = self.interactive_document_to_json(current_meeting_record)
                with open(os.path.join(meetings_path, "meeting_record.json"), "w", encoding="utf-8") as f:
                    json.dump(json_content, f, indent=2, ensure_ascii=False)

            # Save agent contexts
            for agent_name, agent_context in current_agent_contexts.items():
                # Create agent folder
                agent_folder = os.path.join(meetings_path, agent_name)
                os.makedirs(agent_folder, exist_ok=True)

                # Save as HTML
                html_content = self.interactive_document_to_html(agent_context)
                with open(os.path.join(agent_folder, "context.html"), "w", encoding="utf-8") as f:
                    f.write(html_content)

                # Save as JSON
                json_content = self.interactive_document_to_json(agent_context)
                with open(os.path.join(agent_folder, "context.json"), "w", encoding="utf-8") as f:
                    json.dump(json_content, f, indent=2, ensure_ascii=False)

            # Save agent summaries if available
            for agent_name, summary in agent_meeting_summaries.items():
                # Create agent folder if it doesn't exist yet
                agent_folder = os.path.join(meetings_path, agent_name)
                os.makedirs(agent_folder, exist_ok=True)

                # Save as HTML
                html_content = self.interactive_document_to_html(summary)
                with open(os.path.join(agent_folder, "summary.html"), "w", encoding="utf-8") as f:
                    f.write(html_content)

                # Save as JSON
                json_content = self.interactive_document_to_json(summary)
                with open(os.path.join(agent_folder, "summary.json"), "w", encoding="utf-8") as f:
                    json.dump(json_content, f, indent=2, ensure_ascii=False)

            # Save voting and position tracking data if available
            if hasattr(meeting_manager, 'voting_manager') and meeting_manager.voting_manager.has_votes(meeting_id):
                votes_data = {
                    "initial": meeting_manager.voting_manager.votes.get(meeting_id, {}).get("initial", {}),
                    "alternative": meeting_manager.voting_manager.votes.get(meeting_id, {}).get("alternative", {})
                }

                # Convert complex objects to serializable format
                def make_serializable(obj):
                    if isinstance(obj, dict):
                        return {k: make_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [make_serializable(item) for item in obj]
                    elif isinstance(obj, (int, float, str, bool, type(None))):
                        return obj
                    else:
                        return str(obj)

                votes_data = make_serializable(votes_data)

                with open(os.path.join(meetings_path, "votes.json"), "w", encoding="utf-8") as f:
                    json.dump(votes_data, f, indent=2, ensure_ascii=False)

            if hasattr(meeting_manager,
                       'position_manager') and meeting_id in meeting_manager.position_manager.positions:
                positions_data = meeting_manager.position_manager.positions.get(meeting_id, {})
                position_dict = {}

                # Convert to serializable format
                for agent_name, positions in positions_data.items():
                    position_dict[agent_name] = [
                        {
                            "round": p.get("round", 0),
                            "option": p.get("option", ""),
                            "reasoning": p.get("reasoning", ""),
                            "changed": p.get("changed", False)
                        }
                        for p in positions
                    ]

                with open(os.path.join(meetings_path, "positions.json"), "w", encoding="utf-8") as f:
                    json.dump(position_dict, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"Error saving meeting results: {e}")
            return False


    def save_simulation_results(self, simulation_folder, company):
        """
        Save final simulation results.

        Args:
            simulation_folder (str): Path to the simulation folder
            company: The company object containing the document manager

        Returns:
            bool: Success status
        """
        documents_path = os.path.join(simulation_folder, "documents")
        os.makedirs(documents_path, exist_ok=True)

        # Print the folder path for debugging
        print(f"Saving simulation results to: {os.path.abspath(documents_path)}")

        success_count = 0
        error_count = 0

        try:
            # Check if document_manager exists
            if not hasattr(company, 'meeting_manager') or not hasattr(company.meeting_manager, 'document_manager'):
                print(f"Warning: Company does not have document_manager attribute")
                return False

            # Get document manager
            document_manager = company.meeting_manager.document_manager

            # Save full record
            if hasattr(document_manager, 'full_record') and document_manager.full_record:
                try:
                    # Save as HTML
                    html_content = self.interactive_document_to_html(document_manager.full_record)
                    full_record_path = os.path.join(documents_path, "full_record.html")
                    with open(full_record_path, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    print(f"Successfully saved: {full_record_path}")
                    success_count += 1
                except Exception as e:
                    print(f"Error saving full record HTML: {e}")
                    error_count += 1

                try:
                    # Save as JSON
                    json_content = self.interactive_document_to_json(document_manager.full_record)
                    json_path = os.path.join(documents_path, "full_record.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(json_content, f, indent=2, ensure_ascii=False)
                    print(f"Successfully saved: {json_path}")
                    success_count += 1
                except Exception as e:
                    print(f"Error saving full record JSON: {e}")
                    error_count += 1

            # Save agent contexts
            for agent_name, agent_context in document_manager.agent_contexts.items():
                # Create agent folder
                agent_folder = os.path.join(documents_path, agent_name)
                os.makedirs(agent_folder, exist_ok=True)

                try:
                    # Save as HTML
                    html_content = self.interactive_document_to_html(agent_context)
                    html_path = os.path.join(agent_folder, "context.html")
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    print(f"Successfully saved: {html_path}")
                    success_count += 1
                except Exception as e:
                    print(f"Error saving agent context HTML: {e}")
                    error_count += 1

                try:
                    # Save as JSON
                    json_content = self.interactive_document_to_json(agent_context)
                    json_path = os.path.join(agent_folder, "context.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(json_content, f, indent=2, ensure_ascii=False)
                    print(f"Successfully saved: {json_path}")
                    success_count += 1
                except Exception as e:
                    print(f"Error saving agent context JSON: {e}")
                    error_count += 1

            # Save agent memories
            for agent_name, agent_memory in document_manager.agent_memories.items():
                # Create agent folder if it doesn't exist yet
                agent_folder = os.path.join(documents_path, agent_name)
                os.makedirs(agent_folder, exist_ok=True)

                try:
                    # Save as HTML
                    html_content = self.interactive_document_to_html(agent_memory)
                    html_path = os.path.join(agent_folder, "memory.html")
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    print(f"Successfully saved: {html_path}")
                    success_count += 1
                except Exception as e:
                    print(f"Error saving agent memory HTML: {e}")
                    error_count += 1

                try:
                    # Save as JSON
                    json_content = self.interactive_document_to_json(agent_memory)
                    json_path = os.path.join(agent_folder, "memory.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(json_content, f, indent=2, ensure_ascii=False)
                    print(f"Successfully saved: {json_path}")
                    success_count += 1
                except Exception as e:
                    print(f"Error saving agent memory JSON: {e}")
                    error_count += 1

            # Save simulation document if available
            if hasattr(company, 'simulation_document'):
                try:
                    html_content = self.interactive_document_to_html(company.simulation_document)
                    html_path = os.path.join(documents_path, "simulation_summary.html")
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    print(f"Successfully saved: {html_path}")
                    success_count += 1
                except Exception as e:
                    print(f"Error saving simulation summary HTML: {e}")
                    error_count += 1

                try:
                    json_content = self.interactive_document_to_json(company.simulation_document)
                    json_path = os.path.join(documents_path, "simulation_summary.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(json_content, f, indent=2, ensure_ascii=False)
                    print(f"Successfully saved: {json_path}")
                    success_count += 1
                except Exception as e:
                    print(f"Error saving simulation summary JSON: {e}")
                    error_count += 1

            print(f"Saving complete: {success_count} files saved successfully, {error_count} errors encountered")
            return success_count > 0

        except Exception as e:
            print(f"Error saving simulation results: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_agent_components(self, simulation_folder, agents_dict, ceo=None):
        """
        Save agent component measurements.

        Args:
            simulation_folder (str): Path to the simulation folder
            agents_dict (dict): Dictionary of shareholders
            ceo: CEO agent (optional)

        Returns:
            bool: Success status
        """
        components_path = os.path.join(simulation_folder, "components")
        os.makedirs(components_path, exist_ok=True)

        try:
            # Save shareholders' components
            for agent_name, agent in agents_dict.items():
                if hasattr(agent, 'measurements'):
                    # Create HTML from measurements
                    html_content = self.measurements_to_html(agent.measurements)
                    with open(os.path.join(components_path, f"{agent_name}_components.html"), "w",
                              encoding="utf-8") as f:
                        f.write(html_content)

            # Save CEO's components if provided
            if ceo and hasattr(ceo, 'measurements'):
                html_content = self.measurements_to_html(ceo.measurements)
                with open(os.path.join(components_path, f"{ceo.config.name}_components.html"), "w",
                          encoding="utf-8") as f:
                    f.write(html_content)

            return True

        except Exception as e:
            print(f"Error saving agent components: {e}")
            return False

    def save_all_simulation_data(self, company, num_agents):
        """
        Complete function to save all simulation data after completion.
        Creates folders and saves all documents, meetings, and component measurements.

        Args:
            company: The company object containing simulation data
            num_agents: Number of agents in the simulation

        Returns:
            str: Path to the simulation folder
        """
        # Create simulation folder
        simulation_folder = self.create_simulation_folder(num_agents)

        # Save final simulation results
        self.save_simulation_results(simulation_folder, company)

        # Save agent components
        self.save_agent_components(
            simulation_folder,
            company.shareholders,
            ceo=company.ceo
        )

        print(f"All simulation data saved to: {simulation_folder}")
        return simulation_folder

    def save_decision_tracker(self, simulation_folder, decision_tracker):
        """
        Save the decision tracker data to the simulation folder.

        Args:
            simulation_folder (str): Path to the simulation folder
            decision_tracker: The DecisionTracker object containing all simulation decisions

        Returns:
            bool: Success status
        """
        try:
            # Ensure the simulation folder exists
            os.makedirs(simulation_folder, exist_ok=True)

            # Save the decision tracker data as JSON
            decision_json_path = os.path.join(simulation_folder, "decision_tracker_data.json")

            # Use the decision tracker's built-in export method
            decision_tracker.export_to_json(decision_json_path)

            print(f"Successfully saved decision tracker to: {decision_json_path}")

            # Optionally, create a human-readable summary
            self._create_decision_tracker_summary(simulation_folder, decision_tracker)

            return True

        except Exception as e:
            print(f"Error saving decision tracker: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_decision_tracker_summary(self, simulation_folder, decision_tracker):
        """
        Create a human-readable HTML summary of the decision tracker data.

        Args:
            simulation_folder (str): Path to the simulation folder
            decision_tracker: The DecisionTracker object
        """
        try:
            # Get complete data
            data = decision_tracker.get_complete_data()

            # Generate summary statistics
            summary_stats = decision_tracker.get_summary_statistics()

            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Decision Tracker Summary</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        padding: 20px;
                        max-width: 1200px;
                        margin: 0 auto;
                        color: #333;
                    }}
                    h1, h2, h3 {{
                        color: #222;
                        margin-top: 20px;
                    }}
                    h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
                    h2 {{ border-bottom: 1px solid #666; padding-bottom: 5px; }}
                    .stat {{
                        margin: 10px 0;
                        padding: 10px;
                        background-color: #f9f9f9;
                        border-left: 4px solid #0066cc;
                    }}
                    .ceo-info, .shareholder-info {{
                        background-color: #f4f4f4;
                        padding: 15px;
                        margin: 10px 0;
                        border-radius: 5px;
                    }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin-top: 20px;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }}
                    th {{
                        background-color: #f2f2f2;
                        font-weight: bold;
                    }}
                    .toc {{
                        background-color: #f8f8f8;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        padding: 15px;
                        margin-bottom: 20px;
                    }}
                    .toc ul {{
                        list-style-type: none;
                        padding-left: 0;
                    }}
                    .toc li {{
                        margin-bottom: 5px;
                    }}
                    .toc a {{
                        text-decoration: none;
                        color: #0066cc;
                    }}
                    .toc a:hover {{
                        text-decoration: underline;
                    }}
                </style>
            </head>
            <body>
                <h1>Decision Tracker Summary</h1>

                <div class="toc">
                    <h2>Table of Contents</h2>
                    <ul>
                        <li><a href="#overview">Overview</a></li>
                        <li><a href="#simulation-setup">Simulation Setup</a></li>
                        <li><a href="#summary-statistics">Summary Statistics</a></li>
                        <li><a href="#ceo-performance">CEO Performance</a></li>
                        <li><a href="#company-performance">Company Performance</a></li>
                        <li><a href="#meeting-summary">Meeting Summary</a></li>
                    </ul>
                </div>

                <h2 id="overview">Overview</h2>
                <div class="stat">
                    Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>

                <h2 id="simulation-setup">Simulation Setup</h2>
                <div class="ceo-info">
                    <h3>CEO</h3>
                    <p><strong>Name:</strong> {data['CEO']['name']}</p>
                    <p><strong>MBTI:</strong> {data['CEO']['MBTI']}</p>
                </div>

                <h3>Shareholders</h3>
            """

            # Add shareholder information
            for shareholder in data.get('company', {}).get('shareholders', []):
                html_content += f"""
                <div class="shareholder-info">
                    <p><strong>Name:</strong> {shareholder['name']}</p>
                    <p><strong>MBTI:</strong> {shareholder['mbti']}</p>
                </div>
                """

            # Add summary statistics
            html_content += f"""
                <h2 id="summary-statistics">Summary Statistics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Meetings</td>
                        <td>{summary_stats.get('total_meetings', 0)}</td>
                    </tr>
                    <tr>
                        <td>Total Discussion Rounds</td>
                        <td>{summary_stats.get('total_rounds', 0)}</td>
                    </tr>
                    <tr>
                        <td>Total Votes Cast</td>
                        <td>{summary_stats.get('total_votes', 0)}</td>
                    </tr>
                    <tr>
                        <td>Initial Assets</td>
                        <td>${summary_stats.get('performance_metrics', {}).get('initial_assets', 0):,.2f}</td>
                    </tr>
                    <tr>
                        <td>Final Assets</td>
                        <td>${summary_stats.get('performance_metrics', {}).get('final_assets', 0):,.2f}</td>
                    </tr>
                    <tr>
                        <td>Growth Rate</td>
                        <td>{summary_stats.get('performance_metrics', {}).get('growth_rate', 0):.2f}%</td>
                    </tr>
                </table>

                <h2 id="ceo-performance">CEO Performance</h2>
            """

            # Add CEO review data if available
            ceo_reviews = data.get('company', {}).get('company_review', {})
            if ceo_reviews:
                html_content += "<h3>Annual Reviews</h3>"
                for year, review in ceo_reviews.items():
                    ceo_ratings = review.get('ceo_ratings', {})
                    if ceo_ratings:
                        html_content += f"""
                        <h4>{year}</h4>
                        <p><strong>Average Rating:</strong> {ceo_ratings.get('average_score', 0):.1f}/5</p>
                        """

            # Add company performance
            html_content += """
                <h2 id="company-performance">Company Performance</h2>
            """

            # Add company review data if available
            if ceo_reviews:
                html_content += "<h3>Annual Reviews</h3>"
                for year, review in ceo_reviews.items():
                    company_ratings = review.get('company_ratings', {})
                    if company_ratings:
                        html_content += f"""
                        <h4>{year}</h4>
                        <p><strong>Average Rating:</strong> {company_ratings.get('average_score', 0):.1f}/5</p>
                        """

            # Add meeting summary
            html_content += """
                <h2 id="meeting-summary">Meeting Summary</h2>
                <p>For detailed meeting information, see the individual meeting files in the meetings folder.</p>

            </body>
            </html>
            """

            # Save the HTML summary
            summary_path = os.path.join(simulation_folder, "decision_tracker_summary.html")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"Successfully saved decision tracker summary to: {summary_path}")

        except Exception as e:
            print(f"Error creating decision tracker summary: {e}")