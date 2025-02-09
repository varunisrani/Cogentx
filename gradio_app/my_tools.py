import os
from utils import rnd_id
from crewai_tools import (CodeInterpreterTool, ScrapeElementFromWebsiteTool, TXTSearchTool, SeleniumScrapingTool,
                         PGSearchTool, PDFSearchTool, MDXSearchTool, JSONSearchTool, GithubSearchTool, EXASearchTool,
                         DOCXSearchTool, CSVSearchTool, ScrapeWebsiteTool, FileReadTool, DirectorySearchTool,
                         DirectoryReadTool, CodeDocsSearchTool, YoutubeVideoSearchTool, SerperDevTool,
                         YoutubeChannelSearchTool, WebsiteSearchTool)
from tools.CSVSearchToolEnhanced import CSVSearchToolEnhanced
from tools.CustomApiTool import CustomApiTool
from tools.CustomCodeInterpreterTool import CustomCodeInterpreterTool
from tools.CustomFileWriteTool import CustomFileWriteTool
from tools.ScrapeWebsiteToolEnhanced import ScrapeWebsiteToolEnhanced
from langchain_community.tools import YahooFinanceNewsTool
import gradio as gr

class MyTool:
    def __init__(self, tool_id, name, description, parameters, **kwargs):
        self.tool_id = tool_id or rnd_id()
        self.name = name
        self.description = description
        self.parameters = kwargs
        self.parameters_metadata = parameters

    def create_tool(self):
        pass

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, **kwargs):
        self.parameters.update(kwargs)

    def get_parameter_names(self):
        return list(self.parameters_metadata.keys())

    def is_parameter_mandatory(self, param_name):
        return self.parameters_metadata.get(param_name, {}).get('mandatory', False)

    def is_valid(self):
        validation_errors = []
        for param_name, metadata in self.parameters_metadata.items():
            if metadata['mandatory'] and not self.parameters.get(param_name):
                validation_errors.append(f"Parameter '{param_name}' is mandatory for tool '{self.name}'")
        return len(validation_errors) == 0, validation_errors

    def to_dict(self):
        """Convert tool to dictionary for display"""
        return {
            "id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "mandatory_params": [k for k, v in self.parameters_metadata.items() if v.get('mandatory')]
        }

class MyScrapeWebsiteTool(MyTool):
    def __init__(self, tool_id=None, website_url=None):
        parameters = {
            'website_url': {'mandatory': False}
        }
        super().__init__(tool_id, 'ScrapeWebsiteTool', "A tool that can be used to read website content.", parameters, website_url=website_url)

    def create_tool(self) -> ScrapeWebsiteTool:
        return ScrapeWebsiteTool(self.parameters.get('website_url') if self.parameters.get('website_url') else None)

class MySerperDevTool(MyTool):
    def __init__(self, tool_id=None, api_key=None):
        parameters = {
            'api_key': {'mandatory': True}
        }
        super().__init__(tool_id, 'SerperDevTool', "A tool for searching the web using Serper.dev API.", parameters, api_key=api_key)

    def create_tool(self):
        return SerperDevTool(self.parameters.get('api_key'))

class MyWebsiteSearchTool(MyTool):
    def __init__(self, tool_id=None, website_url=None):
        parameters = {
            'website_url': {'mandatory': True}
        }
        super().__init__(tool_id, 'WebsiteSearchTool', "A tool for searching content on a specific website.", parameters, website_url=website_url)

    def create_tool(self):
        return WebsiteSearchTool(self.parameters.get('website_url'))

class MyScrapeWebsiteToolEnhanced(MyTool):
    def __init__(self, tool_id=None, website_url=None):
        parameters = {
            'website_url': {'mandatory': True}
        }
        super().__init__(tool_id, 'ScrapeWebsiteToolEnhanced', "Enhanced tool for scraping website content.", parameters, website_url=website_url)

    def create_tool(self):
        return ScrapeWebsiteToolEnhanced(self.parameters.get('website_url'))

class MySeleniumScrapingTool(MyTool):
    def __init__(self, tool_id=None, website_url=None):
        parameters = {
            'website_url': {'mandatory': True}
        }
        super().__init__(tool_id, 'SeleniumScrapingTool', "Tool for scraping dynamic websites using Selenium.", parameters, website_url=website_url)

    def create_tool(self):
        return SeleniumScrapingTool(self.parameters.get('website_url'))

class MyScrapeElementFromWebsiteTool(MyTool):
    def __init__(self, tool_id=None, website_url=None, element_xpath=None):
        parameters = {
            'website_url': {'mandatory': True},
            'element_xpath': {'mandatory': True}
        }
        super().__init__(tool_id, 'ScrapeElementFromWebsiteTool', "Tool for scraping specific elements from a website.", 
                        parameters, website_url=website_url, element_xpath=element_xpath)

    def create_tool(self):
        return ScrapeElementFromWebsiteTool(self.parameters.get('website_url'), self.parameters.get('element_xpath'))

class MyCustomApiTool(MyTool):
    def __init__(self, tool_id=None, api_url=None, method=None):
        parameters = {
            'api_url': {'mandatory': True},
            'method': {'mandatory': True}
        }
        super().__init__(tool_id, 'CustomApiTool', "Tool for making custom API calls.", 
                        parameters, api_url=api_url, method=method)

    def create_tool(self):
        return CustomApiTool(self.parameters.get('api_url'), self.parameters.get('method'))

class MyCodeInterpreterTool(MyTool):
    def __init__(self, tool_id=None):
        parameters = {}
        super().__init__(tool_id, 'CodeInterpreterTool', "Tool for interpreting and executing code.", parameters)

    def create_tool(self):
        return CodeInterpreterTool()

class MyCustomCodeInterpreterTool(MyTool):
    def __init__(self, tool_id=None, code=None):
        parameters = {
            'code': {'mandatory': True}
        }
        super().__init__(tool_id, 'CustomCodeInterpreterTool', "Tool for executing custom code.", parameters, code=code)

    def create_tool(self):
        return CustomCodeInterpreterTool(self.parameters.get('code'))

class MyFileReadTool(MyTool):
    def __init__(self, tool_id=None, file_path=None):
        parameters = {
            'file_path': {'mandatory': True}
        }
        super().__init__(tool_id, 'FileReadTool', "Tool for reading file contents.", parameters, file_path=file_path)

    def create_tool(self):
        return FileReadTool(self.parameters.get('file_path'))

class MyCustomFileWriteTool(MyTool):
    def __init__(self, tool_id=None, file_path=None, content=None):
        parameters = {
            'file_path': {'mandatory': True},
            'content': {'mandatory': True}
        }
        super().__init__(tool_id, 'CustomFileWriteTool', "Tool for writing content to files.", 
                        parameters, file_path=file_path, content=content)

    def create_tool(self):
        return CustomFileWriteTool(self.parameters.get('file_path'), self.parameters.get('content'))

class MyDirectorySearchTool(MyTool):
    def __init__(self, tool_id=None, directory_path=None):
        parameters = {
            'directory_path': {'mandatory': True}
        }
        super().__init__(tool_id, 'DirectorySearchTool', "Tool for searching directories.", parameters, directory_path=directory_path)

    def create_tool(self):
        return DirectorySearchTool(self.parameters.get('directory_path'))

class MyDirectoryReadTool(MyTool):
    def __init__(self, tool_id=None, directory_path=None):
        parameters = {
            'directory_path': {'mandatory': True}
        }
        super().__init__(tool_id, 'DirectoryReadTool', "Tool for reading directory contents.", parameters, directory_path=directory_path)

    def create_tool(self):
        return DirectoryReadTool(self.parameters.get('directory_path'))

class MyYoutubeVideoSearchTool(MyTool):
    def __init__(self, tool_id=None, video_url=None):
        parameters = {
            'video_url': {'mandatory': True}
        }
        super().__init__(tool_id, 'YoutubeVideoSearchTool', "Tool for searching YouTube video content.", parameters, video_url=video_url)

    def create_tool(self):
        return YoutubeVideoSearchTool(self.parameters.get('video_url'))

class MyYoutubeChannelSearchTool(MyTool):
    def __init__(self, tool_id=None, channel_url=None):
        parameters = {
            'channel_url': {'mandatory': True}
        }
        super().__init__(tool_id, 'YoutubeChannelSearchTool', "Tool for searching YouTube channel content.", parameters, channel_url=channel_url)

    def create_tool(self):
        return YoutubeChannelSearchTool(self.parameters.get('channel_url'))

class MyGithubSearchTool(MyTool):
    def __init__(self, tool_id=None, repo_url=None):
        parameters = {
            'repo_url': {'mandatory': True}
        }
        super().__init__(tool_id, 'GithubSearchTool', "Tool for searching GitHub repositories.", parameters, repo_url=repo_url)

    def create_tool(self):
        return GithubSearchTool(self.parameters.get('repo_url'))

class MyCodeDocsSearchTool(MyTool):
    def __init__(self, tool_id=None, docs_url=None):
        parameters = {
            'docs_url': {'mandatory': True}
        }
        super().__init__(tool_id, 'CodeDocsSearchTool', "Tool for searching code documentation.", parameters, docs_url=docs_url)

    def create_tool(self):
        return CodeDocsSearchTool(self.parameters.get('docs_url'))

class MyYahooFinanceNewsTool(MyTool):
    def __init__(self, tool_id=None):
        parameters = {}
        super().__init__(tool_id, 'YahooFinanceNewsTool', "Tool for fetching Yahoo Finance news.", parameters)

    def create_tool(self):
        return YahooFinanceNewsTool()

class MyTXTSearchTool(MyTool):
    def __init__(self, tool_id=None, file_path=None):
        parameters = {
            'file_path': {'mandatory': True}
        }
        super().__init__(tool_id, 'TXTSearchTool', "Tool for searching text files.", parameters, file_path=file_path)

    def create_tool(self):
        return TXTSearchTool(self.parameters.get('file_path'))

class MyCSVSearchTool(MyTool):
    def __init__(self, tool_id=None, file_path=None):
        parameters = {
            'file_path': {'mandatory': True}
        }
        super().__init__(tool_id, 'CSVSearchTool', "Tool for searching CSV files.", parameters, file_path=file_path)

    def create_tool(self):
        return CSVSearchTool(self.parameters.get('file_path'))

class MyCSVSearchToolEnhanced(MyTool):
    def __init__(self, tool_id=None, file_path=None):
        parameters = {
            'file_path': {'mandatory': True}
        }
        super().__init__(tool_id, 'CSVSearchToolEnhanced', "Enhanced tool for searching CSV files.", parameters, file_path=file_path)

    def create_tool(self):
        return CSVSearchToolEnhanced(self.parameters.get('file_path'))

class MyDocxSearchTool(MyTool):
    def __init__(self, tool_id=None, file_path=None):
        parameters = {
            'file_path': {'mandatory': True}
        }
        super().__init__(tool_id, 'DOCXSearchTool', "Tool for searching DOCX files.", parameters, file_path=file_path)

    def create_tool(self):
        return DOCXSearchTool(self.parameters.get('file_path'))

class MyEXASearchTool(MyTool):
    def __init__(self, tool_id=None, file_path=None):
        parameters = {
            'file_path': {'mandatory': True}
        }
        super().__init__(tool_id, 'EXASearchTool', "Tool for searching EXA files.", parameters, file_path=file_path)

    def create_tool(self):
        return EXASearchTool(self.parameters.get('file_path'))

class MyJSONSearchTool(MyTool):
    def __init__(self, tool_id=None, file_path=None):
        parameters = {
            'file_path': {'mandatory': True}
        }
        super().__init__(tool_id, 'JSONSearchTool', "Tool for searching JSON files.", parameters, file_path=file_path)

    def create_tool(self):
        return JSONSearchTool(self.parameters.get('file_path'))

class MyMDXSearchTool(MyTool):
    def __init__(self, tool_id=None, file_path=None):
        parameters = {
            'file_path': {'mandatory': True}
        }
        super().__init__(tool_id, 'MDXSearchTool', "Tool for searching MDX files.", parameters, file_path=file_path)

    def create_tool(self):
        return MDXSearchTool(self.parameters.get('file_path'))

class MyPDFSearchTool(MyTool):
    def __init__(self, tool_id=None, file_path=None):
        parameters = {
            'file_path': {'mandatory': True}
        }
        super().__init__(tool_id, 'PDFSearchTool', "Tool for searching PDF files.", parameters, file_path=file_path)

    def create_tool(self):
        return PDFSearchTool(self.parameters.get('file_path'))

# Dictionary mapping tool names to their respective classes
TOOL_CLASSES = {
    'ScrapeWebsiteTool': MyScrapeWebsiteTool,
    'SerperDevTool': MySerperDevTool,
    'WebsiteSearchTool': MyWebsiteSearchTool,
    'ScrapeWebsiteToolEnhanced': MyScrapeWebsiteToolEnhanced,
    'SeleniumScrapingTool': MySeleniumScrapingTool,
    'ScrapeElementFromWebsiteTool': MyScrapeElementFromWebsiteTool,
    'CustomApiTool': MyCustomApiTool,
    'CodeInterpreterTool': MyCodeInterpreterTool,
    'CustomCodeInterpreterTool': MyCustomCodeInterpreterTool,
    'FileReadTool': MyFileReadTool,
    'CustomFileWriteTool': MyCustomFileWriteTool,
    'DirectorySearchTool': MyDirectorySearchTool,
    'DirectoryReadTool': MyDirectoryReadTool,
    'YoutubeVideoSearchTool': MyYoutubeVideoSearchTool,
    'YoutubeChannelSearchTool': MyYoutubeChannelSearchTool,
    'GithubSearchTool': MyGithubSearchTool,
    'CodeDocsSearchTool': MyCodeDocsSearchTool,
    'YahooFinanceNewsTool': MyYahooFinanceNewsTool,
    'TXTSearchTool': MyTXTSearchTool,
    'CSVSearchTool': MyCSVSearchTool,
    'CSVSearchToolEnhanced': MyCSVSearchToolEnhanced,
    'DOCXSearchTool': MyDocxSearchTool,
    'EXASearchTool': MyEXASearchTool,
    'JSONSearchTool': MyJSONSearchTool,
    'MDXSearchTool': MyMDXSearchTool,
    'PDFSearchTool': MyPDFSearchTool,
}
