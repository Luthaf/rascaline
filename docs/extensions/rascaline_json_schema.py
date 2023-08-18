import json
import os

from docutils import nodes
from docutils.parsers.rst import Directive
from html_hidden import html_hidden
from markdown_it import MarkdownIt
from myst_parser.config.main import MdParserConfig
from myst_parser.mdit_to_docutils.base import DocutilsRenderer


def markdow_to_docutils(text):
    parser = MarkdownIt()
    tokens = parser.parse(text)

    renderer = DocutilsRenderer(parser)
    return renderer.render(tokens, {"myst_config": MdParserConfig()}, {})


def _target_id(text):
    return nodes.make_id("json-schema-" + text)


class JsonSchemaDirective(Directive):
    required_arguments = 1

    def __init__(self, *args, **kwargs):
        super(JsonSchemaDirective, self).__init__(*args, **kwargs)
        self._inline_call_count = 0
        self.docs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def run(self, *args, **kwargs):
        schema, content = self._load_schema()

        title = f"{schema['title']} hyper-parameters"
        root_target, section = self._transform(schema, title)

        schema_node = html_hidden(toggle="Show full JSON schema")
        schema_node += nodes.literal_block(text=content)
        section.insert(1, schema_node)

        for name, definition in schema.get("definitions", {}).items():
            target, subsection = self._transform(definition, name)

            section += target
            section += subsection

        return [root_target, section]

    def _transform(self, schema, name):
        target_id = _target_id(name)
        target = nodes.target(
            "", "", ids=[target_id], names=[target_id], line=self.lineno
        )

        section = nodes.section()
        section += nodes.title(text=name)

        description = schema.get("description", "")
        section.extend(markdow_to_docutils(description))

        section += self._json_schema_to_nodes(schema)

        return (target, section)

    def _load_schema(self):
        path = os.path.join(self.docs_root, self.arguments[0])
        if not os.path.exists(path):
            raise Exception(f"Unable to find JSON schema at '{path}'.")

        self.state.document.settings.env.note_dependency(path)

        with open(path) as fd:
            content = fd.read()

        schema = json.loads(content)

        schema["$$rust-type"] = schema["title"]
        schema["title"] = os.path.basename(path).split(".")[0]

        return schema, content

    def _json_schema_to_nodes(self, schema, inline=False):
        """Transform the schema for a single type to docutils nodes"""

        if "type" in schema:
            if schema["type"] == "object":
                if not inline:
                    field_list = nodes.field_list()
                    for name, content in schema.get("properties", {}).items():
                        name = nodes.field_name(text=name)
                        name += nodes.Text(": ")
                        if "default" in content:
                            name += nodes.Text("optional, ")

                        name += self._json_schema_to_nodes(content, inline=True)

                        field_list += name
                        body = nodes.field_body()

                        description = content.get("description", "")
                        body.extend(markdow_to_docutils(description))

                        field_list += body

                    return field_list
                else:
                    self._inline_call_count += 1

                    object_node = nodes.inline()

                    if self._inline_call_count > 1:
                        object_node += nodes.Text("{")

                    for name, content in schema.get("properties", {}).items():
                        field = nodes.inline()
                        field += nodes.Text(f"{name}: ")

                        subfields = self._json_schema_to_nodes(content, inline=True)
                        if isinstance(subfields, nodes.literal):
                            subfields = [subfields]

                        for i, sf in enumerate(subfields):
                            field += sf

                            if isinstance(sf, nodes.inline):
                                if i != len(subfields) - 2:
                                    # len(xxx) - 2 to account for the final }
                                    field += nodes.Text(", ")
                        object_node += field

                    if self._inline_call_count > 1:
                        object_node += nodes.Text("}")

                    self._inline_call_count -= 1
                    return object_node

            elif schema["type"] == "number":
                assert schema["format"] == "double"
                return nodes.literal(text="number")

            elif schema["type"] == "integer":
                if "format" not in schema:
                    return nodes.literal(text="integer")

                if schema["format"].startswith("int"):
                    return nodes.literal(text="integer")
                elif schema["format"].startswith("uint"):
                    return nodes.literal(text="unsigned integer")
                else:
                    raise Exception(f"unknown integer format: {schema['format']}")

            elif schema["type"] == "string":
                if "enum" in schema:
                    values = [f'"{v}"' for v in schema["enum"]]
                    return nodes.literal(text=" | ".join(values))
                else:
                    return nodes.literal(text="string")

            elif schema["type"] == "boolean":
                return nodes.literal(text="boolean")

            elif isinstance(schema["type"], list):
                # we only support list for Option<T>
                assert len(schema["type"]) == 2
                assert schema["type"][1] == "null"

                schema["type"] = schema["type"][0]
                return self._json_schema_to_nodes(schema, inline=True)

            elif schema["type"] == "array":
                array_node = nodes.inline()
                array_node += self._json_schema_to_nodes(schema["items"], inline=True)
                array_node += nodes.Text("[]")
                return array_node

            else:
                raise Exception(f"unsupported JSON type ({schema['type']}) in schema")

        if "$ref" in schema:
            ref = schema["$ref"]
            assert ref.startswith("#/definitions/")
            type_name = ref.split("/")[-1]

            refid = _target_id(type_name)

            return nodes.reference(internal=True, refid=refid, text=type_name)

        # enums values are represented as allOf
        if "allOf" in schema:
            assert len(schema["allOf"]) == 1
            return self._json_schema_to_nodes(schema["allOf"][0])

        # Enum variants uses "oneOf"
        if "oneOf" in schema:
            bullet_list = nodes.bullet_list()
            for possibility in schema["oneOf"]:
                item = nodes.list_item()
                item += self._json_schema_to_nodes(possibility, inline=True)

                description = possibility.get("description", "")
                item.extend(markdow_to_docutils(description))

                bullet_list += item

            return bullet_list

        raise Exception(f"unsupported JSON schema: {schema}")


def setup(app):
    app.require_sphinx("3.3")
    app.add_directive("rascaline-json-schema", JsonSchemaDirective)
